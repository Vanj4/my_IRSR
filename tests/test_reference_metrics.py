import unittest
import numpy as np
import torch
from utils.psnr_ssim_calculate import reference_metrics, calculate_psnr, calculate_ssim

class ReferenceMetricChecks(unittest.TestCase):
    def test_known_constant_error(self):
        a=np.full((40,40),0.4);b=np.full((40,40),0.5)
        self.assertAlmostEqual(calculate_psnr(a,b),20.,places=10)
        self.assertAlmostEqual(calculate_ssim(a,b),(2*.4*.5+.01**2)/(.4**2+.5**2+.01**2),places=10)
    def test_exact_match_and_border(self):
        a=np.full((40,40),0.5);b=a.copy();b[:4]=0
        self.assertEqual(reference_metrics(a,b,4),{'psnr':160.,'ssim':1.})
        self.assertLess(calculate_psnr(a,b,0),160.)
        self.assertEqual(reference_metrics(torch.tensor(a)[None,None],torch.tensor(a)[None,None]),{'psnr':160.,'ssim':1.})
    def test_reject_invalid_inputs(self):
        a=np.ones((40,40))
        for b,border in [(a,-1),(a,True),(a[:20],4),(a*2,4),(a*float('nan'),4),(a,16)]:
            with self.assertRaises(ValueError):reference_metrics(a,b,border)
