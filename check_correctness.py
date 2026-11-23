"""Small independent checks for scientific failure modes before real training."""
import numpy as np
import torch
from common import PALETTE, confusion, decode_mask, metrics, save_json, EVIDENCE
from models import UNet, segmentation_loss

def main():
    torch.set_num_threads(2)
    y = np.array([[0,1,1],[2,2,0]], dtype=np.uint8)
    assert np.array_equal(decode_mask(PALETTE[y]), y)
    try: decode_mask(np.array([[[1,2,3]]], dtype=np.uint8))
    except ValueError: pass
    else: raise AssertionError('Unknown colors must fail')
    pred = np.array([[0,1,0],[2,1,0]])
    cm = confusion(y,pred)
    assert cm.tolist() == [[2,0,0],[1,1,0],[0,1,1]]
    m = metrics(cm)
    assert abs(m['iou'][1]-1/3)<1e-12 and abs(m['iou'][2]-.5)<1e-12
    assert metrics(np.diag([10,0,0]))['foreground_miou'] is None
    target = torch.tensor(y.astype(np.int64)).unsqueeze(0)
    perfect = torch.nn.functional.one_hot(target,3).permute(0,3,1,2).float()*20
    assert segmentation_loss(perfect,target).item()<1e-5
    torch.manual_seed(42); base = UNet()
    torch.manual_seed(42); attn = UNet(attention=True)
    assert all(torch.equal(v,attn.state_dict()[k]) for k,v in base.state_dict().items())
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    results = []
    for model in [base,attn]:
        model = model.to(device)
        x = torch.randn(2,6,256,256,device=device)
        y = torch.randint(0,3,(2,256,256),device=device)
        opt = torch.optim.AdamW(model.parameters(),lr=3e-4)
        before = model.output.weight.detach().clone()
        logits = model(x)
        assert logits.shape == (2,3,256,256)
        loss = segmentation_loss(logits,y)
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if hasattr(model.attention,'block'):
            assert model.attention.block.self_attn.in_proj_weight.grad.abs().sum().item()>0
        opt.step()
        assert not torch.equal(before,model.output.weight)
        if device.type=='mps': torch.mps.synchronize()
        results.append({'model':'attention' if hasattr(model.attention,'block') else 'baseline',
                        'parameters':sum(p.numel() for p in model.parameters()),'loss':loss.item(),
                        'forward_backward_update':'passed'})
        model.cpu()
        del x,y,logits,loss,opt
        if device.type=='mps': torch.mps.empty_cache()
    save_json(EVIDENCE/'correctness-checks.json',{'metric_handcheck':'passed',
        'color_mapping':'passed','unknown_color_rejected':True,'loss_perfect_prediction':'passed',
        'matched_common_initialization':True,'device':str(device),'models':results})
    print(results,flush=True)

if __name__ == '__main__': main()
