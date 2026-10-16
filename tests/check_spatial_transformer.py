"""Focused regression checks for spatial refinement; run from project root."""
import sys,importlib.util,json,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import config
from network import FrequencyCrossAttentionSR
from spatial_transformer import SpatialTransformerLayer
from utils.runtime import setup,Engine
from dataset import InfraredSRDataset
from torch.utils.data import default_collate

setup()
out=config.RESULT_ROOT/'spatial_transformer_checks'
out.mkdir(parents=True,exist_ok=True)
report={}
old_path=config.PROJECT_ROOT/'_backups/before_spatial_transformer_20261014/network.py'
spec=importlib.util.spec_from_file_location('old_network',old_path)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
for rgb in (False,True):
    torch.manual_seed(19);a=old.FrequencyCrossAttentionSR(use_rgb=rgb).eval()
    torch.manual_seed(19);b=FrequencyCrossAttentionSR(use_rgb=rgb,spatial_groups=0).eval()
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    lr=torch.rand(1,1,13,17); guide=torch.rand(1,3,52,68) if rgb else None
    with torch.no_grad(): assert torch.equal(a(lr,guide),b(lr,guide))
    torch.manual_seed(19);c=FrequencyCrossAttentionSR(use_rgb=rgb)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    report['parameters_rgb_'+str(rgb)]=sum(p.numel() for p in c.parameters())
    del a,b,c
report['disabled_exact_original_and_initialization']=True
# Chunked attention matches unchunked output; checkpoint matches output and gradients.
l=SpatialTransformerLayer(16,4,8,4,2,3).eval()
x=torch.randn(2,16,16,24,requires_grad=True)
y=l(x); l.eval_chunk=999; z=l(x)
assert torch.allclose(y,z,atol=1e-6,rtol=1e-5)
report['chunk_max_error']=float((y-z).detach().abs().max())
labels=l.region_labels(16,24,4,4,'cpu')
assert (labels[0,:,None]==labels[0,None,:]).all()
assert torch.unique(labels[-1]).numel()==4
# A far right edge cannot wrap into the left edge through shifted attention.
x=torch.randn(1,16,16,24);xx=x.clone();xx[:,:,:,-1]+=100
with torch.no_grad(): ya=l(x);yb=l(xx)
assert torch.equal(ya[:,:,:,0],yb[:,:,:,0])
report['no_cyclic_boundary_leak']=True
m=FrequencyCrossAttentionSR(channels=16,num_blocks=1,use_rgb=False)
x=torch.rand(1,1,13,17)
y=m(x);y.square().mean().backward()
grads={n:p.grad.clone() for n,p in m.named_parameters()}
m.zero_grad(set_to_none=True)
for group in m.spatial_refinement: group.use_checkpoint=False
z=m(x);z.square().mean().backward()
assert torch.equal(y,z)
assert all(torch.equal(g,m.get_parameter(n).grad) for n,g in grads.items())
assert z.shape==(1,1,52,68)
report['checkpoint_output_gradient_exact']=True
for h,w in [(1,1),(8,16),(9,19)]:
    with torch.no_grad(): assert m(torch.rand(1,1,h,w)).shape==(1,1,4*h,4*w)
report['small_odd_rectangular_shapes']=True
del m,x,y,z,l;gc.collect()
# Actual 96-channel RGB model, FP16 optimizer step, EMA and exact resumed next step.
setup()
data=InfraredSRDataset(config.TRAIN_ROOT,seed=config.SEED)
batches=[default_collate([data[i]]) for i in range(2)]
engine=Engine(4,1)
stats=engine.update([batches[0]])
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in engine.model.spatial_refinement.parameters())
report['amp_step']=stats
report['spatial_gradients_finite_nonzero']=all(p.grad.abs().sum().item()>0 for p in engine.model.spatial_refinement.parameters())
assert report['spatial_gradients_finite_nonzero']
contract={'check':'spatial_refinement_v1'}
engine.save(out/'resume.pt',{'epoch':0,'offset':1},contract)
stats2=engine.update([batches[1]])
expected={k:v.detach().cpu().clone() for k,v in engine.model.state_dict().items()}
expected_ema={k:v.detach().cpu().clone() for k,v in engine.ema.module.state_dict().items()}
del engine;gc.collect();torch.cuda.empty_cache()
engine=Engine(4,1)
assert engine.restore(out/'resume.pt',contract)=={'epoch':0,'offset':1}
stats3=engine.update([batches[1]])
assert stats2==stats3
assert all(torch.equal(v,engine.model.state_dict()[k].cpu()) for k,v in expected.items())
assert all(torch.equal(v,engine.ema.module.state_dict()[k].cpu()) for k,v in expected_ema.items())
report['next_step_resume_model_ema_exact']=True
report['train_peak_GiB']=torch.cuda.max_memory_allocated()/2**30
# Real full-resolution sample, same whole-image protocol used in validation.
del batches,expected,expected_ema;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
full=InfraredSRDataset(config.VALIDATION_ROOT,hr_patch_size=None,random_crop=False,augment=False)[0]
with torch.no_grad():
    sr=engine.ema.module(full['lr'].unsqueeze(0).cuda(),full['rgb'].unsqueeze(0).cuda())
assert sr.shape==(1,1,*full['hr'].shape[-2:]) and torch.isfinite(sr).all()
report['full_image_shape']=list(sr.shape)
report['full_image_peak_GiB']=torch.cuda.max_memory_allocated()/2**30
(out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
