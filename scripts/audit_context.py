"""Read-only checkpoint diagnosis. No training/model-source mutation."""
import sys,json,copy,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
import config
from network import FrequencyCrossAttentionSR,haar_dwt,haar_iwt
from dataset import build_dataset
from utils.runtime import setup
from utils.psnr_ssim_calculate import reference_metrics

OUT=config.RESULT_ROOT/'audit_context_v1'

def array(x):return x.detach().float().cpu()

def trace(model,x):
    result={};handles=[];size=x.shape[-1]
    def record(name,t):
        if isinstance(t,(tuple,list)):t=torch.cat(t,dim=0)
        n=t.shape[-1]*32//size;y=(t.shape[-2]-n)//2;z=(t.shape[-1]-n)//2
        result[name]=array(t[...,y:y+n,z:z+n])
    def hook(name):
        def f(mod,args,out):
            if name.endswith('cross_attention'):
                record(name+'.low',args[0]);record(name+'.high',args[1])
                record(name+'.delta',tuple(o-h for o,h in zip(out,args[1])))
            record(name,out)
        return f
    for name,mod in model.named_modules():
        if name in ('stem','body_conv','upsample','head') or (name.startswith('blocks.') and (name.count('.')==1 or name.endswith(('low_encoder','high_encoder','cross_attention','fuse')))):
            handles.append(mod.register_forward_hook(hook(name)))
    try:
        with torch.inference_mode():pred=model(x)
    finally:
        for h in handles:h.remove()
    record('SR',pred)
    return pred,result

def diff(a,b):
    n=a.shape[-1];c=n*3//8;d=n//4
    a,b=a[...,c:c+d,c:c+d],b[...,c:c+d,c:c+d]
    return {'mae':float((a-b).abs().mean()),'max':float((a-b).abs().max()),'relative_l2':float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))}

def main():
    setup();OUT.mkdir(exist_ok=True)
    ds=build_dataset('validation');indices=np.linspace(0,len(ds)-1,20,dtype=int).tolist()
    manifest=[str(ds.paths[i]) for i in indices];(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];layers=[];alphas={}
    for ck in ('best','last'):
        state=torch.load(config.CHECKPOINT_ROOT/config.RUN_NAME/(ck+'.pt'),map_location='cpu',weights_only=False)
        for name,digest in state['contract']['source'].items():assert hashlib.sha256((config.PROJECT_ROOT/name).read_bytes()).hexdigest()==digest,name
        model=FrequencyCrossAttentionSR().cuda().eval();model.load_state_dict(state['ema']['model'])
        alphas[ck]=[float(b.cross_attention.alpha.detach()) for b in model.blocks]
        for order,idx in enumerate(indices):
            sample=ds[idx];lr=sample['lr'][None].cuda();hr=sample['hr'][None]
            h,w=lr.shape[-2:];y=((h-128)//16)*8;x=((w-128)//16)*8
            large=lr[...,y:y+128,x:x+128];small=large[...,48:80,48:80]
            gt=hr[...,(y+48)*4:(y+80)*4,(x+48)*4:(x+80)*4]
            assert y%8==x%8==0
            ps,ts=trace(model,small);pl,tl=trace(model,large)
            for name in ts:layers.append({'checkpoint':ck,'image':ds.paths[idx].name,'layer':name,**diff(ts[name],tl[name])})
            def add(mode,a,b):
                a=a.float().clamp(0,1).cpu();b=b[...,192:320,192:320].float().clamp(0,1).cpu()
                rows.append({'checkpoint':ck,'image':ds.paths[idx].name,'mode':mode,
                  'small':reference_metrics(a,gt,4),'large':reference_metrics(b,gt,4),
                  'core_gap':diff(a,b), 'sr_mae':float((a-b).abs().mean())})
            add('original',ps,pl)
            with torch.inference_mode():
                # Diagnostic interventions on copies/in-memory settings only; not retrained comparisons.
                saved=[b.cross_attention.alpha.clone() for b in model.blocks]
                for b in model.blocks:b.cross_attention.alpha.zero_()
                add('cross_update_off',model(small),model(large))
                for b,v in zip(model.blocks,saved):b.cross_attention.alpha.copy_(v)
                for mod in model.modules():
                    if isinstance(mod,nn.Conv2d):mod.padding_mode='reflect'
                add('reflect_all_convs',model(small),model(large))
                for mod in model.modules():
                    if isinstance(mod,nn.Conv2d):mod.padding_mode='zeros'
            if order%5==0:print(ck,order+1,'/20',flush=True)
        del model
    (OUT/'layer_differences.json').write_text(json.dumps(layers,indent=2))
    (OUT/'interventions.json').write_text(json.dumps(rows,indent=2))
    summary={'sample_count':20,'sampling':'20 uniformly spaced sorted validation indices; fixed central regions','alphas':alphas,'groups':[],'layers':[]}
    for ck in ('best','last'):
        for mode in ('original','cross_update_off','reflect_all_convs'):
            rs=[r for r in rows if r['checkpoint']==ck and r['mode']==mode]
            g={'checkpoint':ck,'mode':mode,'small_psnr':np.mean([r['small']['psnr'] for r in rs]),'large_psnr':np.mean([r['large']['psnr'] for r in rs]),'small_ssim':np.mean([r['small']['ssim'] for r in rs]),'large_ssim':np.mean([r['large']['ssim'] for r in rs]),'core_mae':np.mean([r['core_gap']['mae'] for r in rs])}
            summary['groups'].append(g);print(g,flush=True)
        for name in dict.fromkeys(r['layer'] for r in layers):
            rs=[r for r in layers if r['checkpoint']==ck and r['layer']==name]
            summary['layers'].append({'checkpoint':ck,'layer':name,**{k:float(np.mean([r[k] for r in rs])) for k in ('mae','max','relative_l2')}})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
