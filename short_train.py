"""Bounded diagnostic/resource runs. These are NOT final validation/test results."""
import argparse
import csv
import json
import math
import platform
import random
import statistics
import threading
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from common import ROOT, EVIDENCE, confusion, metrics, save_json, source_hashes
from dataset import PairDataset, manifest, select_diagnostic
from models import UNet, segmentation_loss

def sync(device):
    if device.type == 'mps': torch.mps.synchronize()

def setup(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    # Matched seeds do not promise bitwise deterministic MPS kernels.

class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak_rss = 0; self.minimum_available = psutil.virtual_memory().available
        self.start_swap = psutil.swap_memory().used; self.peak_swap = self.start_swap
        def sample():
            while not self.stop.wait(.2):
                self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
                self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)
                self.peak_swap = max(self.peak_swap, psutil.swap_memory().used)
        self.thread = threading.Thread(target=sample,daemon=True); self.thread.start()
    def finish(self):
        self.stop.set(); self.thread.join()
        return {'process_peak_rss_bytes_sampled':self.peak_rss,
            'system_minimum_available_bytes_sampled':self.minimum_available,
            'system_swap_start_bytes':self.start_swap,'system_peak_swap_bytes_sampled':self.peak_swap,
            'sampling_interval_seconds':.2,
            'note':'RSS, system memory, and MPS allocations overlap; do not sum them'}

@torch.no_grad()
def evaluate(model, dataset, device, batch_size=2):
    model.eval(); cm = np.zeros((3,3),dtype=np.int64); loss_sum = 0
    started = time.perf_counter()
    for x,y in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
        x,y = x.to(device),y.to(device)
        logits = model(x)
        loss_sum += segmentation_loss(logits,y).item()*len(x)
        cm += confusion(y.cpu().numpy(),logits.argmax(1).cpu().numpy())
    sync(device)
    return {'loss':loss_sum/len(dataset),**metrics(cm),'seconds':time.perf_counter()-started}

def run(args):
    if args.mode == 'diagnostic':
        path = EVIDENCE/'diagnostic-samples.json'
        names = json.loads(path.read_text())['filenames'] if path.exists() else select_diagnostic()
        ds = PairDataset('train',names,cache=True)
    else:
        names = [r['filename'] for r in manifest('train')]
        ds = PairDataset('train',names,augment=True,seed=args.seed)
    setup(args.seed)
    device = torch.device(args.device)
    out = ROOT/'runs'/f'{args.mode}_{args.model}_seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    config = {**vars(args),'data_split':'train','filenames':names,
        'normalization':json.loads((EVIDENCE/'normalization.json').read_text()),
        'loss':'0.5 CE + 0.5 foreground Dice, batch pooled, smooth 1e-6; absent class FP penalized',
        'attention':{'heads':4,'ffn':512,'dropout':.1,'position':'fixed 2D sinusoidal'},
        'source_sha256':source_hashes(),'data_source':json.loads((EVIDENCE/'data-source.json').read_text()),
        'torch':torch.__version__,'python':platform.python_version(),'platform':platform.platform(),
        'status':'diagnostic only; no held-out accuracy claim'}
    save_json(out/'config.json',config)
    model = UNet(attention=args.model=='attention',width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    checkpoint = out/'last.pt'
    start_step, history = 0, []
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['config']['source_sha256'] != config['source_sha256']:
            raise ValueError('Code changed since checkpoint: review before resuming')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type=='mps' and saved.get('mps_rng') is not None: torch.mps.set_rng_state(saved['mps_rng'])
        start_step, history = saved['step'], saved['history']
    elif checkpoint.exists():
        raise ValueError('Run exists; use --resume or a different seed')
    monitor = MemoryMonitor(); times=[]; loading_times=[]; mps_peak=0; driver_peak=0
    started = time.perf_counter()
    if args.mode=='diagnostic' and not history:
        initial = evaluate(model,ds,device,args.batch)
        history.append({'step':0,'train_batch_loss':None,'diagnostic':initial})
        print('initial',json.dumps(initial),flush=True)
    log = (out/'steps.jsonl').open('a' if args.resume else 'w', buffering=1)
    # Precomputed per-epoch order and index-based geometry isolate data RNG from dropout.
    epoch, loader_iter = None,None
    try:
        for step in range(start_step,args.steps):
            batches_per_epoch = math.ceil(len(ds)/args.batch)
            next_epoch, batch_index = divmod(step,batches_per_epoch)
            if next_epoch != epoch:
                epoch = next_epoch; ds.epoch=epoch
                generator = torch.Generator().manual_seed(args.seed+epoch)
                loader_iter = iter(DataLoader(ds,batch_size=args.batch,shuffle=True,
                    generator=generator,num_workers=0))
                for _ in range(batch_index): next(loader_iter)
            before_io = time.perf_counter(); x,y = next(loader_iter)
            loading = time.perf_counter()-before_io
            model.train(); sync(device); before_step=time.perf_counter()
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = segmentation_loss(logits,y)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Non-finite loss')
            loss.backward(); optimizer.step(); sync(device)
            elapsed=time.perf_counter()-before_step
            if step >= args.warmup:
                times.append(elapsed); loading_times.append(loading)
            if device.type=='mps':
                mps_peak=max(mps_peak,torch.mps.current_allocated_memory())
                driver_peak=max(driver_peak,torch.mps.driver_allocated_memory())
            row={'step':step+1,'loss':loss.item(),'compute_seconds':elapsed,'loading_seconds':loading}
            log.write(json.dumps(row)+'\n')
            if (step+1)%20==0: print(json.dumps(row),flush=True)
            if (step+1)%args.evaluate_every==0 or step+1==args.steps:
                entry={'step':step+1,'train_batch_loss':loss.item()}
                if args.mode=='diagnostic': entry['diagnostic']=evaluate(model,ds,device,args.batch)
                history.append(entry)
                save_json(out/'history.json',history)
                saved={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'step':step+1,'history':history,'config':config,'torch_rng':torch.get_rng_state(),
                    'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None}
                torch.save(saved,out/'last.tmp'); (out/'last.tmp').replace(checkpoint)
                print('checkpoint',step+1,json.dumps(entry),flush=True)
            del x,y,logits,loss
    finally:
        log.close()
        memory = monitor.finish()
        save_json(out/'memory.json',{**memory,'mps_allocated_peak_bytes_at_step_boundaries':mps_peak,
                                    'mps_driver_peak_bytes_at_step_boundaries':driver_peak})
    duration=time.perf_counter()-started
    timing={'steps_measured':len(times),'warmup_steps_excluded':args.warmup,
        'mean_compute_seconds':statistics.mean(times) if times else None,
        'median_compute_seconds':statistics.median(times) if times else None,
        'p90_compute_seconds':float(np.percentile(times,90)) if times else None,
        'mean_io_seconds':statistics.mean(loading_times) if times else None,
        'first_half_mean':statistics.mean(times[:len(times)//2]) if len(times)>1 else None,
        'second_half_mean':statistics.mean(times[len(times)//2:]) if len(times)>1 else None,
        'elapsed_seconds_including_diagnostic_evaluation':duration}
    inference=None
    if args.mode=='benchmark':
        model.eval(); x,y=ds[0]; x=x.unsqueeze(0).to(device)
        latency=[]
        with torch.no_grad():
            for i in range(70):
                sync(device); t=time.perf_counter(); model(x); sync(device)
                if i>=20: latency.append(time.perf_counter()-t)
        inference={'batch':1,'warmup':20,'repetitions':50,
            'mean_seconds':statistics.mean(latency),'median_seconds':statistics.median(latency),
            'excludes':'file reading, rendering, serialization', 'input':'one real training pair'}
        # Validation time only: this diagnostic checkpoint never selects the final model.
        val_names=[r['filename'] for r in manifest('val')][:64]
        if val_names:
            val=evaluate(model,PairDataset('val',val_names),device,args.batch)
            inference['validation_64_pair_seconds']=val['seconds']
        else:
            inference['validation_timing_status']='pending full data; pilot has train only'
    summary={'mode':args.mode,'model':args.model,'steps':args.steps,
        'parameters':sum(p.numel() for p in model.parameters()),'timing':timing,
        'memory':json.loads((out/'memory.json').read_text()),'inference':inference,
        'initial_diagnostic':history[0].get('diagnostic'),
        'final_diagnostic':history[-1].get('diagnostic'),
        'warning':'Training-subset fit/throughput only; NOT validation/test performance; NOT evidence attention improves generalization'}
    save_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['diagnostic','benchmark'],required=True)
    p.add_argument('--model',choices=['baseline','attention'],required=True)
    p.add_argument('--steps',type=int,default=320); p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--batch',type=int,default=2); p.add_argument('--width',type=int,default=16)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='mps'); p.add_argument('--evaluate-every',type=int,default=40)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
