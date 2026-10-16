"""Train infrared SR with optional aligned HR-RGB guidance. Run with the project .venv Python; see V1_IMPLEMENTATION.md."""
import argparse, atexit, hashlib, json, math, time, msvcrt
from pathlib import Path
import torch
import config
from dataset import InfraredSRDataset, build_dataset
from utils.runtime import setup, configuration, fingerprint, epoch_loader, Engine
from utils.evaluation import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--spatial-groups',type=int,default=config.SPATIAL_GROUPS,help='Residual spatial groups; 0 disables refinement')
    parser.add_argument('--rgb-guidance',action=argparse.BooleanOptionalAction,default=config.USE_RGB_GUIDANCE)
    parser.add_argument('--smoke',action='store_true',help='8 train / 2 full validation images, 2 epochs; isolated outputs')
    parser.add_argument('--run-name',default=config.RUN_NAME)
    parser.add_argument('--channels',type=int,default=config.CHANNELS,help='Feature width for this run')
    parser.add_argument('--num-blocks',type=int,default=config.NUM_BLOCKS,help='Frequency interaction blocks for this run')
    parser.add_argument('--hr-patch-size',type=int,default=config.HR_PATCH_SIZE,help='HR crop size; overrides config for this run only')
    parser.add_argument('--max-updates',type=int,help='Stop/save after this many successful updates, without changing schedule')
    args=parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.','..'): parser.error('Invalid run name')
    if args.max_updates is not None and args.max_updates<1: parser.error('max-updates must be positive')
    if args.hr_patch_size < config.SCALE or args.hr_patch_size % config.SCALE:
        parser.error('hr-patch-size must be positive and divisible by SCALE')
    if args.channels < 1 or args.channels % config.ATTENTION_HEADS:
        parser.error('channels must be positive and divisible by ATTENTION_HEADS')
    if args.num_blocks < 1:
        parser.error('num-blocks must be positive')
    if args.spatial_groups<0: parser.error('spatial-groups must be nonnegative')
    config.SPATIAL_GROUPS = args.spatial_groups
    config.USE_RGB_GUIDANCE = args.rgb_guidance
    config.CHANNELS = args.channels
    config.NUM_BLOCKS = args.num_blocks
    config.HR_PATCH_SIZE = args.hr_patch_size
    config.RUN_NAME = args.run_name
    setup()
    data=InfraredSRDataset(config.TRAIN_ROOT,seed=config.SEED)
    val=build_dataset('validation')
    epochs=config.EPOCHS
    validate_every=config.VALIDATE_EVERY_EPOCHS
    save_every=config.SAVE_EVERY_EPOCHS
    name=args.run_name
    if args.smoke:
        data.paths=data.paths[:8];val.paths=val.paths[:2]
        epochs=2;validate_every=save_every=1;name+='__smoke'
    if min(epochs,config.BATCH_SIZE,config.ACCUMULATION_STEPS,validate_every,save_every)<1: raise ValueError('Invalid training counts')
    directory=config.CHECKPOINT_ROOT/name
    output=config.RESULT_ROOT/name
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'run.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise RuntimeError('This run is already active; do not launch twice')
    atexit.register(lock.close)
    if directory.exists() and any(directory.glob('*.pt')) and args.resume is None:
        raise FileExistsError(f'{directory} already has weights; use --resume or another run name')
    directory.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    train_files,val_files=fingerprint(data),fingerprint(val)
    if {r['sha256'] for r in train_files} & {r['sha256'] for r in val_files}:
        raise ValueError('Identical file content occurs in train and validation')
    if config.USE_RGB_GUIDANCE and ({r['rgb_sha256'] for r in train_files} & {r['rgb_sha256'] for r in val_files}):
        raise ValueError('Identical RGB file content occurs in train and validation')
    total=epochs*math.ceil(math.ceil(len(data)/config.BATCH_SIZE)/config.ACCUMULATION_STEPS)
    warmup=min(config.WARMUP_UPDATES,total-1) if args.smoke else config.WARMUP_UPDATES
    source_paths=['network.py','spatial_transformer.py','rgb_high_frequency_encoder.py','dataset.py','scripts/prepare_data.py','losses/sr_loss.py',
                  'utils/runtime.py','utils/training_components.py','utils/evaluation.py',
                  'utils/brisque_niqe_piqe_calculate.py','utils/psnr_ssim_calculate.py','train.py']
    contract={'best_criterion':'psnr_ssim_joint_historical_max',
              'config':configuration(),'epochs':epochs,'total_updates':total,'warmup_updates':warmup,
              'train':train_files,'validation':val_files,'smoke':args.smoke,
              'source':{p:hashlib.sha256((config.PROJECT_ROOT/p).read_bytes()).hexdigest() for p in source_paths}}
    engine=Engine(total,warmup)
    position={'epoch':0,'offset':0}
    if args.resume: position=engine.restore(args.resume,contract)
    (output/'resolved_run.json').write_text(json.dumps(contract,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[SPATIAL] groups={config.SPATIAL_GROUPS} depth={config.SPATIAL_DEPTH} window={config.SPATIAL_WINDOW} heads={config.SPATIAL_HEADS} parameters={sum(p.numel() for p in engine.model.parameters())}',flush=True)
    print(f'[GUIDE] rgb={config.USE_RGB_GUIDANCE} | gate_init={config.RGB_GATE_INIT} | RGB encoder tile={config.RGB_GUIDE_TILE}',flush=True)
    print(f'[START] train={len(data)} validation={len(val)} epochs={epochs} batch={config.BATCH_SIZE} '
          f'accumulation={config.ACCUMULATION_STEPS} amp={config.AMP} updates={engine.updates}/{total}',flush=True)
    print(f'[PROTOCOL] full-image EMA; best=PSNR and SSIM strict historical maxima; last every {save_every} epochs',flush=True)
    if engine.updates>=total:
        print('[DONE] checkpoint already completed',flush=True);return
    start=time.perf_counter();initial=engine.updates;values=[]
    logpath=output/'train.jsonl'
    def log(kind,**kwargs):
        row={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'kind':kind,'step':engine.updates,**kwargs}
        with logpath.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(row['time'],f'[{kind}]', ' | '.join(f'{k}={v}' for k,v in row.items() if k not in ('time','kind')),flush=True)
    def save(name):
        engine.save(directory/name,dict(position),contract);log('SAVE',path=str(directory/name))
    for epoch in range(position['epoch'],epochs):
        offset=position['offset'] if epoch==position['epoch'] else 0
        loader=epoch_loader(data,epoch,offset)
        group=[]
        for index,batch in enumerate(loader):
            group.append(batch)
            if len(group)<config.ACCUMULATION_STEPS and index+1<len(loader):continue
            stats=engine.update(group);offset+=sum(b['hr'].shape[0] for b in group);group=[]
            position={'epoch':epoch,'offset':offset}
            values.append(stats['loss'])
            if engine.updates%config.PRINT_EVERY==0 or engine.updates==initial+1:
                elapsed=time.perf_counter()-start
                log('TRAIN',epoch=round(epoch+offset/len(data),3),loss=round(stats['loss'],7),
                    mean_loss=round(sum(values)/len(values),7),lr=stats['lr'],
                    grad_preclip=round(stats['grad_preclip'],4),amp_skips=engine.skipped,
                    elapsed_seconds=round(elapsed,1),eta_seconds=round(elapsed/(engine.updates-initial)*(total-engine.updates)),
                    peak_alloc_GiB=round(torch.cuda.max_memory_allocated()/2**30,3) if engine.device.type=='cuda' else 0)
                values=[]
            if args.max_updates and engine.updates>=args.max_updates and offset<len(data):
                save('last.pt');log('STOP',reason='max-updates');return
        position={'epoch':epoch+1,'offset':0}
        if (epoch+1)%validate_every==0 or epoch+1==epochs:
            log('VAL START',epoch=epoch+1,images=len(val),weights='ema')
            # Preserve a recovery point if evaluation fails, without selecting invalid best.
            try:
                scores=evaluate(engine.ema.module,val,output/f'validation_epoch_{epoch+1:04d}')
            except Exception:
                engine.save(directory/'evaluation_failed.pt',{'epoch':epoch,'offset':len(data)},contract)
                raise
            better=engine.best.consider({n:scores[n] for n in engine.best.names},epoch+1)
            log('VAL',epoch=epoch+1,**scores)
            log('BEST',updated=better,selected=engine.best.selected,historical_maxima=engine.best.maxima)
            if better:save('best.pt')
        if (epoch+1)%save_every==0 or epoch+1==epochs:save('last.pt')
        if args.max_updates and engine.updates>=args.max_updates:
            if (epoch+1)%save_every:save('last.pt')
            log('STOP',reason='max-updates');return
    log('DONE',epochs=epochs,best=engine.best.selected)


if __name__=='__main__':main()
