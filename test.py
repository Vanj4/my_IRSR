"""Whole-image EMA inference and per-image quality export using checkpoint protocol."""
import argparse, hashlib
from pathlib import Path
import torch
import config
from dataset import InfraredSRDataset
from network import FrequencyCrossAttentionSR
from utils.runtime import setup
from utils.evaluation import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,help='HR images for synthetic bicubic SR evaluation; defaults to checkpoint validation set')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--with-nr',action='store_true',help='Also calculate BRISQUE/NIQE/PIQE for post-training evaluation')
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('limit must be positive')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    for name in ('network.py','spatial_transformer.py','rgb_high_frequency_encoder.py','dataset.py','scripts/prepare_data.py','utils/evaluation.py','utils/brisque_niqe_piqe_calculate.py','utils/psnr_ssim_calculate.py'):
        if hashlib.sha256((config.PROJECT_ROOT/name).read_bytes()).hexdigest() != state['contract']['source'][name]:
            raise ValueError(f'Evaluation source differs from checkpoint: {name}')
    saved=state['contract']['config']
    # Architecture, normalization, degradation, metric and EMA contracts follow training.
    for name in ('SCALE','DATA_RANGE','CHANNELS','NUM_BLOCKS','ATTENTION_HEADS','WINDOW_SIZE','CROSS_ALPHA_INIT','USE_RGB_GUIDANCE','RGB_GATE_INIT','RGB_GUIDE_TILE','EVAL_BORDER','IQA_DEVICE','RECURSIVE'):
        setattr(config,name,saved[name])
    for name in ('SPATIAL_GROUPS','SPATIAL_DEPTH','SPATIAL_HEADS','SPATIAL_WINDOW','SPATIAL_MLP_RATIO','SPATIAL_RESIDUAL_SCALE','SPATIAL_CHECKPOINT','SPATIAL_EVAL_CHUNK'):
        setattr(config,name,saved[name])
    setup()
    model=FrequencyCrossAttentionSR().to(config.DEVICE)
    model.load_state_dict(state['ema']['model'],strict=True)
    data=InfraredSRDataset(args.input or saved['VALIDATION_ROOT'],hr_patch_size=None,random_crop=False,augment=False)
    if args.limit:data.paths=data.paths[:args.limit]
    scores=evaluate(model,data,args.output,export_all=True,include_nr=args.with_nr)
    print({'checkpoint':str(args.checkpoint),'weights':'ema','step':state['updates'],**scores},flush=True)


if __name__=='__main__':main()
