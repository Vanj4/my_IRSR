import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
import config
from dataset import InfraredSRDataset, build_dataset
from network import FrequencyCrossAttentionSR
from losses import SRLoss
from scripts import prepare_data
from utils.training_components import build_adamw, WarmupCosine, ModelEMA

class ConfigLayoutChecks(unittest.TestCase):
    def test_config_changes_reach_consumers(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            cv2.imwrite(str(Path(root)/"ir.png"),np.full((48,48),100,np.uint8))
            with patch.multiple(config,TRAIN_ROOT=Path(root),HR_PATCH_SIZE=32,
                 CHANNELS=8,NUM_BLOCKS=1,ATTENTION_HEADS=2,WINDOW_SIZE=2,
                 CROSS_ALPHA_INIT=0.2,HORIZONTAL_FLIP_PROBABILITY=0.0,
                 ROTATION_ANGLES=(180,),LEARNING_RATE=0.001,WEIGHT_DECAY=0.02,
                 TOTAL_UPDATES=20,WARMUP_UPDATES=2,MIN_LEARNING_RATE=0.00001,
                 EMA_DECAY=0.9,L1_WEIGHT=0.8):
                data=build_dataset();sample=data[0]
                self.assertEqual(sample['hr'].shape,(1,32,32))
                self.assertEqual(sample['lr'].shape,(1,8,8))
                self.assertEqual(data.rotation_angles,(180,))
                model=FrequencyCrossAttentionSR()
                self.assertEqual(model.stem.out_channels,8)
                self.assertEqual(len(model.blocks),1)
                self.assertAlmostEqual(float(model.blocks[0].cross_attention.alpha.detach()),0.2)
                optimizer=build_adamw(model)
                self.assertEqual(optimizer.param_groups[0]['lr'],0.001)
                self.assertEqual(optimizer.param_groups[0]['weight_decay'],0.02)
                schedule=WarmupCosine(optimizer)
                self.assertAlmostEqual(schedule.set_for_update(1)[0],0.0005)
                self.assertAlmostEqual(schedule.set_for_update(20)[0],0.00001)
                self.assertEqual(ModelEMA(model).decay,0.9)
                self.assertEqual(SRLoss().l1_weight,0.8)
                self.assertIsNone(InfraredSRDataset(root,hr_patch_size=None).hr_patch_size)
                self.assertEqual(SRLoss(l1_weight=1).l1_weight,1)

    def test_module_boundaries_and_unset_validation(self):
        self.assertFalse(hasattr(prepare_data,'InfraredSRDataset'))
        self.assertFalse(hasattr(prepare_data,'read_infrared_image'))
        with patch.object(config,'VALIDATION_ROOT',None):
            with self.assertRaisesRegex(ValueError,'config.py'):build_dataset('validation')
