"""Reproducible training/checkpoint primitives shared by train and tests."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib, json, random
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import config
from network import FrequencyCrossAttentionSR
from losses import SRLoss
from utils.training_components import build_adamw, WarmupCosine, ModelEMA


def setup(seed=config.SEED):
    torch.set_num_threads(config.CPU_THREADS)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cache = config.PROJECT_ROOT / '.cache'
    os.environ['TORCH_HOME'] = str(cache / 'torch')
    os.environ['HF_HOME'] = str(cache / 'huggingface')
    os.environ['XDG_CACHE_HOME'] = str(cache)


def versions():
    out = {}
    for name in ("torch","numpy","opencv-python-headless","pyiqa","scipy"):
        try: out[name] = version(name)
        except PackageNotFoundError: out[name] = None
    return out


def configuration():
    return {k: str(v) if isinstance(v, Path) else v
            for k, v in vars(config).items() if k.isupper()}


def fingerprint(dataset):
    records = []
    for p in dataset.paths:
        record={'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        if config.USE_RGB_GUIDANCE:
            rgb=dataset.rgb_path_for(p)
            record.update(rgb_path=str(rgb.resolve()),rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest())
        records.append(record)
    return records


def epoch_loader(dataset, epoch, offset=0):
    # Local generators avoid consuming model/global RNG when reconstructing a loader.
    dataset.epoch = epoch
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config.SEED + epoch)).tolist()
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=order[offset:],
                      num_workers=config.NUM_WORKERS, pin_memory=config.DEVICE.startswith('cuda'),
                      generator=torch.Generator().manual_seed(config.SEED + epoch + 7919), drop_last=False)


class BestTracker:
    """Strict simultaneous historical maxima of validation mean PSNR and SSIM."""
    names = ('psnr', 'ssim')
    def __init__(self):
        self.maxima = {n: float('-inf') for n in self.names}
        self.selected = None
    def consider(self, scores, epoch):
        if not all(np.isfinite(scores[n]) for n in self.names):
            raise ValueError('Invalid quality scores cannot select best')
        improved = all(scores[n] > self.maxima[n] for n in self.names)
        self.maxima = {n: max(self.maxima[n], scores[n]) for n in self.names}
        if improved: self.selected = {'epoch': epoch, 'scores': dict(scores)}
        return improved
    def state_dict(self): return {'criterion':'psnr_ssim_joint_historical_max','maxima': self.maxima, 'selected': self.selected}
    def load_state_dict(self, state):
        if state.get('criterion')!='psnr_ssim_joint_historical_max': raise ValueError('Best selection criterion mismatch')
        self.maxima, self.selected = state['maxima'], state['selected']


class Engine:
    def __init__(self, total_updates, warmup_updates=None):
        if config.AMP not in ('fp16','off'): raise ValueError('AMP must be fp16 or off')
        if min(config.L1_WEIGHT,config.GRADIENT_WEIGHT,config.EDGE_WEIGHT) < 0 or sum((config.L1_WEIGHT,config.GRADIENT_WEIGHT,config.EDGE_WEIGHT)) <= 0:
            raise ValueError('At least one positive loss weight required')
        self.device = torch.device(config.DEVICE)
        self.model = FrequencyCrossAttentionSR().to(self.device)
        self.optimizer = build_adamw(self.model)
        self.scheduler = WarmupCosine(self.optimizer, total_updates, warmup_updates)
        self.ema = ModelEMA(self.model)
        self.amp = config.AMP == 'fp16' and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.loss = SRLoss().to(self.device)
        self.updates = 0
        self.skipped = 0
        self.best = BestTracker()

    def update(self, batches):
        batches = [(b['lr'].to(self.device, non_blocking=True), b['hr'].to(self.device, non_blocking=True),
                    b['rgb'].to(self.device, non_blocking=True) if self.model.use_rgb else None) for b in batches]
        count = sum(h.shape[0] for _, h, _ in batches)
        self.model.train()
        self.scheduler.set_for_update(self.updates + 1)
        for attempt in range(config.MAX_OVERFLOW_RETRIES):
            self.optimizer.zero_grad(set_to_none=True)
            value = 0.
            for lr, hr, rgb in batches:
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                    sr = self.model(lr,rgb)
                loss = self.loss(sr.float(), hr.float()) * (hr.shape[0] / count)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                value += loss.detach().item()
                self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
            before = self.scaler.get_scale()
            if not self.amp and not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            self.scaler.step(self.optimizer); self.scaler.update()
            if self.scaler.get_scale() >= before:
                self.updates += 1
                self.ema.update(self.model)
                return {'loss': value, 'grad_preclip': float(norm), 'lr': self.optimizer.param_groups[0]['lr']}
            self.skipped += 1
            print(f'[AMP SKIP] retry={attempt+1}, scale={self.scaler.get_scale()}, successful={self.updates}', flush=True)
        raise FloatingPointError('Repeated AMP overflow; no successful update')

    def save(self, path, position, contract):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        state = {'format_version': 1, 'model': self.model.state_dict(), 'ema': self.ema.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                 'scaler': self.scaler.state_dict(), 'updates': self.updates, 'amp_skipped': self.skipped,
                 'best': self.best.state_dict(), 'position': position, 'contract': contract,
                 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                         'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                 'versions': versions(),
                 'evaluation_weights': 'ema'}
        temporary = path.with_suffix('.pt.tmp')
        torch.save(state, temporary); os.replace(temporary, path)

    def restore(self, path, contract):
        # Only load a trusted checkpoint produced by this project: RNG contains numpy state.
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['contract'] != contract: raise ValueError('Resume contract changed (configuration/data/source); start a new run instead')
        if state['versions'] != versions(): raise ValueError('Runtime library versions changed during resume')
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.ema.load_state_dict(state['ema']); self.scaler.load_state_dict(state['scaler'])
        self.updates, self.skipped = state['updates'], state['amp_skipped']
        if self.updates != self.ema.num_updates or self.updates != self.scheduler.last_update:
            raise ValueError('Checkpoint counters disagree')
        self.best.load_state_dict(state['best'])
        random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if state['rng']['cuda']: torch.cuda.set_rng_state_all(state['rng']['cuda'])
        return state['position']
