"""单张灰度图PSNR/SSIM：固定[0,1]量程，裁边后计算，无逐图归一化。
SSIM使用11x11高斯窗口(sigma=1.5)，仅统计有效窗口；与原评价公式一致。
输入支持HW、1HW、11HW的numpy数组或torch张量；不自动clamp或量化。
"""
import math
import cv2
import numpy as np
import torch


def _image(x):
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    x=np.asarray(x)
    if x.ndim==4 and x.shape[:2]==(1,1): x=x[0,0]
    elif x.ndim==3 and x.shape[0]==1: x=x[0]
    if x.ndim!=2 or not x.size: raise ValueError('Expected one grayscale image')
    if not np.isfinite(x).all() or x.min()<0 or x.max()>1: raise ValueError('Expected finite [0,1] pixels')
    return x.astype(np.float64)


def reference_metrics(sr, hr, border=4):
    if not isinstance(border,int) or isinstance(border,bool) or border<0: raise ValueError('Invalid border')
    a,b=_image(sr),_image(hr)
    if a.shape!=b.shape: raise ValueError('SR/HR shape mismatch')
    if min(a.shape)<2*border+11: raise ValueError('Image too small after border crop')
    if border: a,b=a[border:-border,border:-border],b[border:-border,border:-border]
    mse=np.mean((a-b)**2)
    blur=lambda x: cv2.GaussianBlur(x,(11,11),1.5)[5:-5,5:-5]
    ma,mb=blur(a),blur(b)
    va,vb,cov=blur(a*a)-ma*ma,blur(b*b)-mb*mb,blur(a*b)-ma*mb
    ssim=((2*ma*mb+0.01**2)*(2*cov+0.03**2)/((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2))).mean()
    # Preserve existing finite 160 dB cap for exact/effectively exact matches.
    return {'psnr':float(-10*math.log10(max(mse,1e-16))),'ssim':float(ssim)}


def calculate_psnr(sr,hr,border=4): return reference_metrics(sr,hr,border)['psnr']
def calculate_ssim(sr,hr,border=4): return reference_metrics(sr,hr,border)['ssim']
