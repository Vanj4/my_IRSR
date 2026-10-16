"""Shared whole-image validation/inference. Scores use FP32, clamped, unquantized SR."""
import csv, json, math, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import config
METRICS = ("brisque", "niqe", "piqe")


from utils.psnr_ssim_calculate import reference_metrics


def png(path, tensor):
    image = (tensor.detach().float().cpu().squeeze().numpy().clip(0,1)*255).round().astype(np.uint8)
    cv2.imencode('.png', image)[1].tofile(str(path))


@torch.inference_mode()
def evaluate(model, dataset, output, evaluator=None, export_all=False, include_nr=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if include_nr and evaluator is None:
        from utils.brisque_niqe_piqe_calculate import InfraredQualityEvaluator
        evaluator = InfraredQualityEvaluator(config.IQA_DEVICE)
    device = next(model.parameters()).device
    was_training=model.training; model.eval()
    rows=[]; start=time.perf_counter()
    try:
        for index in range(len(dataset)):
            item=dataset[index]
            lr=item['lr'].unsqueeze(0).to(device); hr=item['hr'].unsqueeze(0)
            # Whole image, no tiling seams; native LR feeds the network.
            rgb=item['rgb'].unsqueeze(0).to(device) if getattr(model,'use_rgb',False) else None
            raw=model(lr.float(),rgb) if getattr(model,'use_rgb',False) else model(lr.float())
            if not torch.isfinite(raw).all(): raise FloatingPointError('Nonfinite SR')
            sr=raw.clamp(0,1).cpu()
            if sr.shape != hr.shape: raise ValueError('SR and HR shapes differ')
            row=(evaluator.evaluate_tensor(sr,item['path']) if include_nr else
                 {'image':item['path'],'height':sr.shape[-2],'width':sr.shape[-1],'status':'ok','error':''})
            row.update(reference_metrics(sr,hr,config.EVAL_BORDER))
            bicubic=F.interpolate(lr.float(),size=hr.shape[-2:],mode='bicubic',align_corners=False,antialias=True).clamp(0,1).cpu()
            row.update({'input_'+k:v for k,v in reference_metrics(bicubic,hr,config.EVAL_BORDER).items()})
            rows.append(row)
            stem=f'{index:04d}_{Path(item["path"]).stem}'
            if export_all or index<config.VISUALIZE_COUNT: png(output/(stem+'_SR.png'),sr)
            if index<config.VISUALIZE_COUNT:
                # Left to right: GT, bicubic, SR, absolute error x4 (fixed display scale).
                png(output/(stem+'_GT_BICUBIC_SR_ERRx4.png'),torch.cat((hr,bicubic,sr,4*(sr-hr).abs()),dim=-1))
            if index==0 or (index+1)%20==0 or index+1==len(dataset):
                print(f'[VAL IMAGE] {index+1}/{len(dataset)} status={row["status"]} PSNR={row["psnr"]:.4f}',flush=True)
    finally:
        model.train(was_training)
    with (output/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    valid=all(row['status']=='ok' for row in rows)
    summary={'count':len(rows),'all_nr_valid':valid if include_nr else None,'seconds':time.perf_counter()-start,
             'protocol':{'inference':'whole_image_fp32','nr_border':0,'reference_border':config.EVAL_BORDER,
                         'quantize':False,'clamp':True,'pyiqa':evaluator.backend_version if include_nr else None,'include_nr':include_nr,
                         'degradation':'pytorch_bicubic_antialias','scale':config.SCALE,
                         'rgb_guidance':getattr(model,'use_rgb',False),
                         'rgb_encoder_tile':config.RGB_GUIDE_TILE if getattr(model,'use_rgb',False) else None},
             **{n:float(np.mean([r[n] for r in rows])) for n in ('psnr','ssim','input_psnr','input_ssim')}}
    if include_nr: summary.update({n:float(np.mean([r[n] for r in rows])) if valid else None for n in METRICS})
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if include_nr and not valid: raise RuntimeError(f'Invalid NR scores; best not selected. Inspect {output / "per_image.csv"}')
    return summary
