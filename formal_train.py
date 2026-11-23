"""Segmented formal training with atomic, mid-epoch restart checkpoints.

Data ordering/augmentation use independent per-epoch generators. last.pt is the
restart source of truth; best.pt is selected only by full validation foreground
mean IoU. Budget stops do not reset optimizer, RNG, history or early-stop state.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ROOT, EVIDENCE, confusion, metrics, save_json
from formal_data import approved_data, FormalDataset
from models import UNet, segmentation_loss
from short_train import MemoryMonitor, sync, setup

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    save_json(tmp,value);os.replace(tmp,path)

def atomic_checkpoint(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'mps':torch.mps.get_rng_state() if device.type=='mps' else None}

def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type=='mps' and state['mps'] is not None:torch.mps.set_rng_state(state['mps'])

def epoch_order(n,seed,epoch):
    return torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch)).tolist()

@torch.no_grad()
def validate(model,dataset,device,batch,progress=None):
    model.eval();cm=np.zeros((3,3),dtype=np.int64);loss_sum=0;seen=0
    started=time.perf_counter()
    loader=DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0,
                      generator=torch.Generator().manual_seed(1000042))
    for index,(x,y) in enumerate(loader):
        x,y=x.to(device),y.to(device);logits=model(x)
        loss=segmentation_loss(logits,y)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Validation loss is not finite')
        loss_sum+=loss.item()*len(x);seen+=len(x)
        cm+=confusion(y.cpu().numpy(),logits.argmax(1).cpu().numpy())
        if progress and (index+1)%100==0:progress(seen)
    sync(device)
    if seen!=len(dataset) or int(cm.sum())!=len(dataset)*256*256:
        raise ValueError('Validation sample/pixel count mismatch')
    return {'loss':loss_sum/seen,**metrics(cm),'samples':seen,'seconds':time.perf_counter()-started}

def core_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['formal_train.py','formal_data.py','models.py','dataset.py','common.py','short_train.py']}

def train(args):
    protocol,names,stats=approved_data()
    setup(protocol['seed']);device=torch.device(args.device)
    if args.device!='mps' and not args.smoke:raise ValueError('Approved formal device is MPS')
    if args.smoke:names={s:v[:8 if s=='train' else 4] for s,v in names.items()}
    out=Path(args.out).resolve() if args.out else ROOT/'runs'/f'formal_{args.model}_seed42'
    out.mkdir(parents=True,exist_ok=True)
    process_lock=(out/'run.lock').open('w');fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={'protocol':protocol,'model':args.model,'device':args.device,'smoke_only':args.smoke,
        'train_filenames':names['train'],'validation_filenames':names['val'],'normalization':stats,
        'source_sha256':core_hashes(),'torch':torch.__version__,'python':platform.python_version(),
        'platform':platform.platform(),'batch':2,'width':16,'seed':42,
        'determinism':'data order/geometry controlled; bitwise MPS reproducibility not promised'}
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    checkpoint=out/'last.pt';config_path=out/'config.json'
    model=UNet(attention=args.model=='attention',width=16).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    state={'next_epoch':0,'next_batch':0,'global_step':0,'epoch_loss_sum':0.,'epoch_samples':0,
        'epoch_train_seconds':0.,'total_train_seconds':0.,'total_validation_seconds':0.,
        'history':[],'best_score':-1.,'best_epoch':None,'bad_epochs':0,'early_stopped':False}
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['config_fingerprint']!=fingerprint:
            raise ValueError('Resume configuration/core source differs; refusing to overwrite the run')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        state=saved['state'];restore_rng(saved['rng'],device)
    elif config_path.exists():
        if json.loads(config_path.read_text())!=config:raise ValueError('Existing config mismatch')
    else:
        atomic_json(config_path,config)
        code=out/'source';code.mkdir(exist_ok=True)
        for name in config['source_sha256']:(code/name).write_bytes((ROOT/name).read_bytes())
    train_ds=FormalDataset('train',names['train'],stats,seed=42,augment=True)
    val_ds=FormalDataset('val',names['val'],stats,seed=42,augment=False)
    batch=2;num_batches=math.ceil(len(train_ds)/batch)
    target=min(args.until_epoch,protocol['eventual_max_epochs'])
    if target<state['next_epoch']:raise ValueError('Requested budget predates checkpoint')
    # Roll back orphan log rows after an abrupt interruption to checkpoint step.
    log_path=out/'steps.jsonl'
    if log_path.exists():
        kept=[]
        for line in log_path.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:break
            if row['global_step']<=state['global_step']:kept.append(line)
        log_path.write_text(''.join(line+'\n' for line in kept))
    atomic_json(out/'history.json',state['history'])
    events=(out/'events.jsonl').open('a',buffering=1)
    events.write(json.dumps({'time':now(),'event':'resume' if checkpoint.exists() else 'start',
        'epoch_completed':state['next_epoch'],'next_batch':state['next_batch'],
        'global_step':state['global_step'],'until_epoch':target})+'\n')
    log=log_path.open('a',buffering=1);monitor=MemoryMonitor();session_start=time.perf_counter()
    mps_allocated=0;mps_driver=0;stop_after=False

    def progress(phase,**extra):
        atomic_json(out/'progress.json',{'updated_beijing':now(),'pid':os.getpid(),'model':args.model,
            'phase':phase,'epoch_completed':state['next_epoch'],'current_epoch':state['next_epoch']+1,
            'next_batch':state['next_batch'],'batches_per_epoch':num_batches,'global_step':state['global_step'],
            'segment_target_epochs':target,'best_epoch':state['best_epoch'],'best_validation_foreground_miou':state['best_score'],
            'elapsed_current_process_seconds':time.perf_counter()-session_start,**extra})

    def save():
        sync(device)
        atomic_checkpoint(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'state':state,'rng':rng_state(device),'config_fingerprint':fingerprint,'config':config,
            'saved_beijing':now()})

    try:
        if not checkpoint.exists():save()
        progress('training')
        while state['next_epoch']<target and not state['early_stopped']:
            epoch=state['next_epoch'];train_ds.epoch=epoch
            order=epoch_order(len(train_ds),42,epoch)
            batches=[order[i:i+batch] for i in range(0,len(order),batch)]
            order_hash=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest()
            # Passing explicit batches avoids decoding already-completed samples on resume.
            loader=DataLoader(train_ds,batch_sampler=batches[state['next_batch']:],num_workers=0,
                generator=torch.Generator().manual_seed(2000042+epoch))
            iterator=iter(loader)
            model.train()
            while state['next_batch']<num_batches:
                started=time.perf_counter();x,y=next(iterator);loaded=time.perf_counter()
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True);logits=model(x);loss=segmentation_loss(logits,y)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Training loss is not finite')
                loss.backward();optimizer.step();sync(device)
                duration=time.perf_counter()-started
                batch_index=state['next_batch'];sample_indices=batches[batch_index]
                state['global_step']+=1;state['next_batch']+=1
                state['epoch_loss_sum']+=loss.item()*len(x);state['epoch_samples']+=len(x)
                state['epoch_train_seconds']+=duration;state['total_train_seconds']+=duration
                row={'epoch':epoch+1,'batch':state['next_batch'],'global_step':state['global_step'],
                    'loss':loss.item(),'seconds_with_io':duration,'io_seconds':loaded-started,
                    'filenames':[names['train'][i] for i in sample_indices]}
                log.write(json.dumps(row)+'\n')
                if device.type=='mps':
                    mps_allocated=max(mps_allocated,torch.mps.current_allocated_memory())
                    mps_driver=max(mps_driver,torch.mps.driver_allocated_memory())
                if state['global_step']%50==0 or state['next_batch']==num_batches:
                    progress('training',last_loss=row['loss'],last_step_seconds=duration)
                    print(json.dumps({'time':now(),**row}),flush=True)
                if state['global_step']%args.checkpoint_every==0:save()
                del x,y,logits,loss
                if args.stop_after_steps and state['global_step']>=args.stop_after_steps:
                    save();progress('controlled_interruption_for_verification');stop_after=True;break
            if stop_after:break
            if state['epoch_samples']!=len(train_ds):raise ValueError('Epoch sample count mismatch')
            save();progress('validating',validation_samples_completed=0)
            validation=validate(model,val_ds,device,batch,
                lambda n:progress('validating',validation_samples_completed=n))
            score=validation['foreground_miou']
            if validation['foreground_valid_classes']!=2 or score is None:raise ValueError('Invalid selection metric')
            entry={'epoch':epoch+1,'global_step':state['global_step'],'training_samples':state['epoch_samples'],
                'train_loss':state['epoch_loss_sum']/state['epoch_samples'],
                'train_seconds':state['epoch_train_seconds'],'train_order_sha256':order_hash,'validation':validation}
            state['history'].append(entry);state['total_validation_seconds']+=validation['seconds']
            improved=score>state['best_score']
            if improved:
                state['best_score']=score;state['best_epoch']=epoch+1;state['bad_epochs']=0
                atomic_checkpoint(out/'best.pt',{'model':model.state_dict(),'config':config,
                    'config_fingerprint':fingerprint,'epoch':epoch+1,'validation':validation,'saved_beijing':now()})
            else:state['bad_epochs']+=1
            state['next_epoch']+=1;state['next_batch']=0;state['epoch_loss_sum']=0.
            state['epoch_samples']=0;state['epoch_train_seconds']=0.
            state['early_stopped']=state['bad_epochs']>=protocol['early_stopping_patience']
            save();atomic_json(out/'history.json',state['history'])
            events.write(json.dumps({'time':now(),'event':'epoch_completed','improved':improved,**entry})+'\n')
            print('EPOCH_COMPLETED',json.dumps(entry),flush=True)
        phase='controlled_interruption_for_verification' if stop_after else 'early_stopped' if state['early_stopped'] else 'segment_finished'
        progress(phase)
        if not stop_after:
            atomic_json(out/'summary.json',{'model':args.model,'seed':42,'epochs_completed':state['next_epoch'],
                'segment_target_epochs':target,'best_epoch':state['best_epoch'],'best_validation_foreground_miou':state['best_score'],
                'parameters':sum(p.numel() for p in model.parameters()),'global_step':state['global_step'],
                'total_train_seconds':state['total_train_seconds'],'total_validation_seconds':state['total_validation_seconds'],
                'early_stopped':state['early_stopped'],'training_pairs':len(train_ds),'validation_pairs':len(val_ds),
                'test_evaluated':False,'smoke_only':args.smoke,'history':state['history'],
                'updated_beijing':now(),'status':'segment finished; later epochs and test still pending'})
    except BaseException as error:
        # Atomic last.pt remains valid, including on hard kill. Do not claim unsaved updates survived.
        events.write(json.dumps({'time':now(),'event':'exception','type':type(error).__name__,'message':str(error)})+'\n')
        progress('interrupted_or_failed',error=str(error));raise
    finally:
        log.close();events.close()
        atomic_json(out/f'memory-session-{time.time_ns()}.json',{**monitor.finish(),
            'mps_current_at_step_boundary_max_bytes':mps_allocated,'mps_driver_at_step_boundary_max_bytes':mps_driver,
            'session_elapsed_seconds':time.perf_counter()-session_start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['baseline','attention'],required=True)
    p.add_argument('--until-epoch',type=int,default=5);p.add_argument('--device',default='mps')
    p.add_argument('--checkpoint-every',type=int,default=100);p.add_argument('--out')
    p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after-steps',type=int)
    args=p.parse_args()
    if args.stop_after_steps and not args.smoke:p.error('Controlled interruption is verification-only')
    train(args)
