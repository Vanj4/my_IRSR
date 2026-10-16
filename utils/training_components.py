"""Explicit optimizer/schedule/EMA components; does not launch training.
Schedule indexes SUCCESSFUL optimizer updates. Set LR before attempting update u.
On an AMP skip do not advance completed_updates or call EMA.update().
"""
import copy
import config
import math
import torch
from torch import nn


def build_adamw(model,lr=None,weight_decay=None,betas=None):
    lr = config.LEARNING_RATE if lr is None else lr
    weight_decay = config.WEIGHT_DECAY if weight_decay is None else weight_decay
    betas = config.ADAM_BETAS if betas is None else betas
    if not math.isfinite(lr) or lr<=0 or not math.isfinite(weight_decay) or weight_decay<0:
        raise ValueError('Invalid AdamW learning rate/weight decay')
    decay,no_decay=[],[]
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        # Biases, LayerNorm, scalar residual strengths and relative biases exempt.
        (no_decay if param.ndim<2 or 'relative_bias' in name else decay).append(param)
    return torch.optim.AdamW([{'params':decay,'weight_decay':weight_decay},
                              {'params':no_decay,'weight_decay':0.0}],lr=lr,betas=betas)


class WarmupCosine:
    """1-based upcoming update: first lr=base_lr/warmup; final lr=min_lr."""
    def __init__(self,optimizer,total_updates=None,warmup_updates=None,min_lr=None):
        total_updates = config.TOTAL_UPDATES if total_updates is None else total_updates
        warmup_updates = config.WARMUP_UPDATES if warmup_updates is None else warmup_updates
        min_lr = config.MIN_LEARNING_RATE if min_lr is None else min_lr
        if (not isinstance(total_updates,int) or total_updates<1 or
            not isinstance(warmup_updates,int) or not 0<=warmup_updates<total_updates):
            raise ValueError('Require 0 <= warmup_updates < total_updates')
        if not math.isfinite(min_lr) or min_lr<0: raise ValueError('Invalid min_lr')
        self.optimizer=optimizer
        self.total_updates,self.warmup_updates=total_updates,warmup_updates
        self.min_lr=min_lr
        self.base_lrs=[g['lr'] for g in optimizer.param_groups]
        if any(min_lr>lr for lr in self.base_lrs): raise ValueError('min_lr exceeds base lr')
        self.last_update=0

    def set_for_update(self,update):
        if not isinstance(update,int) or not 1<=update<=self.total_updates:
            raise ValueError('update must be in [1,total_updates]')
        rates=[]
        for base in self.base_lrs:
            if self.warmup_updates and update<=self.warmup_updates:
                lr=base*update/self.warmup_updates
            else:
                start=max(1,self.warmup_updates)
                span=self.total_updates-start
                progress=(update-start)/span if span else 1.0
                lr=self.min_lr+(base-self.min_lr)*0.5*(1+math.cos(math.pi*progress))
            rates.append(lr)
        for g,lr in zip(self.optimizer.param_groups,rates): g['lr']=lr
        self.last_update=update
        return rates

    def state_dict(self):
        return {k:getattr(self,k) for k in ('total_updates','warmup_updates','min_lr','base_lrs','last_update')}

    def load_state_dict(self,state):
        for k in ('total_updates','warmup_updates','min_lr','base_lrs'):
            if state[k]!=getattr(self,k): raise ValueError(f'Scheduler contract changed: {k}')
        if state['last_update']: self.set_for_update(state['last_update'])
        else: self.last_update=0


class ModelEMA:
    """FP32 frozen copy; evaluate explicitly via ema.module. No automatic swap."""
    def __init__(self,model,decay=None):
        decay = config.EMA_DECAY if decay is None else decay
        if not math.isfinite(decay) or not 0<=decay<1: raise ValueError('EMA decay must be in [0,1)')
        self.decay,self.num_updates=float(decay),0
        self.module=copy.deepcopy(model).float().eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self,model):
        src=dict(model.named_parameters())
        dst=dict(self.module.named_parameters())
        if src.keys()!=dst.keys(): raise ValueError('EMA model architecture mismatch')
        for name,p in dst.items():
            q=src[name].detach().to(device=p.device,dtype=p.dtype)
            if p.shape!=q.shape: raise ValueError(f'EMA parameter shape mismatch: {name}')
            p.lerp_(q,1-self.decay)
        buffers=dict(model.named_buffers())
        for name,b in self.module.named_buffers():
            b.copy_(buffers[name].detach().to(device=b.device,dtype=b.dtype))
        self.num_updates+=1
        self.module.eval()

    def state_dict(self):
        return {'decay':self.decay,'num_updates':self.num_updates,'model':self.module.state_dict()}

    def load_state_dict(self,state):
        if state['decay']!=self.decay: raise ValueError('EMA decay changed during resume')
        self.module.load_state_dict(state['model'],strict=True)
        self.num_updates=int(state['num_updates'])
        self.module.eval()
