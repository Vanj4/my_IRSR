"""Run from project root: python -m unittest discover -s tests -p test_frequency_v1.py -v"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch import nn
from network import FrequencyCrossAttentionSR,haar_dwt,haar_iwt
from dataset import InfraredSRDataset,read_infrared_image
from scripts.prepare_data import bicubic_degradation
from utils.training_components import build_adamw,WarmupCosine,ModelEMA
from losses import SRLoss

class V1Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_wavelet_inverse_energy_and_gradient(self):
        x=torch.randn(2,3,18,22,dtype=torch.double,requires_grad=True)
        bands=haar_dwt(x)
        torch.testing.assert_close(haar_iwt(bands),x)
        torch.testing.assert_close(sum(b.square().sum() for b in bands),x.square().sum())
        haar_iwt(bands).sum().backward()
        torch.testing.assert_close(x.grad,torch.ones_like(x))
        constant=torch.ones(1,2,8,8)
        self.assertTrue(all(torch.count_nonzero(b)==0 for b in haar_dwt(constant)[1:]))

    def test_network_shapes_and_all_branches_receive_gradients(self):
        torch.manual_seed(17)
        m=FrequencyCrossAttentionSR(channels=8,num_blocks=1,heads=2)
        x=torch.rand(1,1,17,21)
        y=m(x)
        self.assertEqual(y.shape,(1,1,68,84))
        y.square().mean().backward()
        for name,param in m.named_parameters():
            self.assertIsNotNone(param.grad,name)
            self.assertTrue(torch.isfinite(param.grad).all(),name)
        self.assertEqual(m(torch.rand(1,1,1,2)).shape,(1,1,4,8))
        with self.assertRaises(ValueError):FrequencyCrossAttentionSR(scale=2)

    def test_output_baseline_and_strict_reload(self):
        m=FrequencyCrossAttentionSR(channels=8,num_blocks=1,heads=2)
        with torch.no_grad(): m.head.weight.zero_();m.head.bias.zero_()
        x=torch.rand(1,1,17,19)
        reference=torch.nn.functional.interpolate(x,scale_factor=4,mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        torch.testing.assert_close(m(x),reference)
        other=copy.deepcopy(m)
        other.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(m(x),other(x))

    def test_paired_augmentation_order_and_validation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            image=np.random.default_rng(9).integers(0,256,(140,156),dtype=np.uint8)
            path=Path(directory)/'ir.png';cv2.imwrite(str(path),image)
            ds=InfraredSRDataset(directory)
            torch.manual_seed(42)
            sample=ds[0];meta=json.loads(sample['sample_meta'])
            hr=read_infrared_image(path)[:,meta['top']:meta['top']+128,meta['left']:meta['left']+128]
            lr=bicubic_degradation(hr,4)
            if meta['horizontal_flip']: lr,hr=lr.flip(-1),hr.flip(-1)
            k=meta['angle_degrees']//90
            torch.testing.assert_close(sample['hr'],torch.rot90(hr,k,(-2,-1)),rtol=0,atol=0)
            torch.testing.assert_close(sample['lr'],torch.rot90(lr,k,(-2,-1)),rtol=0,atol=0)
            self.assertEqual(sample['lr'].shape,(1,32,32))
            torch.manual_seed(42)
            self.assertEqual(ds[0]['sample_meta'],sample['sample_meta'])
            valid=InfraredSRDataset(directory,random_crop=False)
            a,b=valid[0],valid[0]
            self.assertEqual(a['sample_meta'],b['sample_meta'])
            self.assertEqual(json.loads(a['sample_meta'])['angle_degrees'],0)
            torch.testing.assert_close(a['hr'],b['hr'])
            full=InfraredSRDataset(directory,hr_patch_size=None)
            self.assertFalse(full.augment)
            for angle in (90,-90,180,-180):
                item=InfraredSRDataset(directory,rotation_angles=(angle,))[0]
                self.assertEqual(json.loads(item['sample_meta'])['angle_degrees'],angle)
            with patch('torch.rand',return_value=torch.tensor(0.249)):
                self.assertTrue(json.loads(ds[0]['sample_meta'])['horizontal_flip'])
            with patch('torch.rand',return_value=torch.tensor(0.25)):
                self.assertFalse(json.loads(ds[0]['sample_meta'])['horizontal_flip'])

    def test_uint16_fixed_range(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'ir16.png'
            cv2.imwrite(str(path),np.full((128,128),4095,dtype=np.uint16))
            torch.testing.assert_close(read_infrared_image(path,4095),torch.ones(1,128,128))
            self.assertLess(read_infrared_image(path).max(),0.1)

    def test_optimizer_schedule_and_state(self):
        model=FrequencyCrossAttentionSR(channels=8,num_blocks=1,heads=2)
        optimizer=build_adamw(model)
        self.assertIsInstance(optimizer,torch.optim.AdamW)
        all_params=[p for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(all_params),len({id(p) for p in model.parameters()}))
        scheduler=WarmupCosine(optimizer)
        self.assertAlmostEqual(scheduler.set_for_update(1)[0],4e-7)
        self.assertAlmostEqual(scheduler.set_for_update(500)[0],2e-4)
        self.assertAlmostEqual(scheduler.set_for_update(50000)[0],1e-6)
        other_optimizer=build_adamw(model)
        other=WarmupCosine(other_optimizer)
        other.load_state_dict(scheduler.state_dict())
        self.assertEqual(other.state_dict(),scheduler.state_dict())
        with self.assertRaises(ValueError):scheduler.set_for_update(50001)

    def test_ema_updates_and_state(self):
        model=nn.Linear(2,2)
        ema=ModelEMA(model,decay=0.5)
        before=ema.module.weight.clone()
        with torch.no_grad():model.weight.add_(2)
        ema.update(model)
        torch.testing.assert_close(ema.module.weight,before+1)
        self.assertTrue(all(not p.requires_grad for p in ema.module.parameters()))
        self.assertFalse(ema.module.training)
        clone=ModelEMA(model,decay=0.5);clone.load_state_dict(ema.state_dict())
        self.assertEqual(clone.num_updates,1)
        torch.testing.assert_close(clone.module.weight,ema.module.weight)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_full_v1_fp16_update_ema_and_eval(self):
        torch.manual_seed(21)
        device='cuda:0'
        model=FrequencyCrossAttentionSR().to(device)
        optimizer=build_adamw(model);scheduler=WarmupCosine(optimizer)
        ema=ModelEMA(model);scaler=torch.amp.GradScaler('cuda')
        lossfn=SRLoss().to(device)
        hr=torch.rand(1,1,128,128,device=device)
        lr=bicubic_degradation(hr,4)
        completed=0
        torch.cuda.reset_peak_memory_stats()
        for attempt in range(10):
            optimizer.zero_grad(set_to_none=True)
            scheduler.set_for_update(completed+1)
            with torch.autocast('cuda',dtype=torch.float16):sr=model(lr)
            loss=lossfn(sr.float(),hr.float())
            self.assertTrue(torch.isfinite(loss))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=previous:
                completed+=1;ema.update(model)
            if completed==2:break
        self.assertEqual(completed,2)
        self.assertEqual(ema.num_updates,2)
        with torch.no_grad():pred=ema.module(lr)
        self.assertEqual(pred.shape,hr.shape)
        self.assertTrue(torch.isfinite(pred).all())
        print('V1_GPU_SMOKE',json.dumps({'parameters':sum(p.numel() for p in model.parameters()),
              'successful_updates':completed,'loss':float(loss.detach()),
              'peak_alloc_gib':torch.cuda.max_memory_allocated()/2**30}))

if __name__=='__main__':unittest.main(verbosity=2)
