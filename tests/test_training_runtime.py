import copy, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
import config
from dataset import InfraredSRDataset
from utils.runtime import setup, Engine, BestTracker, epoch_loader


class TrainingChecks(unittest.TestCase):
    def test_strict_historical_best(self):
        b=BestTracker()
        self.assertTrue(b.consider(dict(psnr=40.,ssim=0.98),1))
        self.assertFalse(b.consider(dict(psnr=41.,ssim=0.97),2))
        self.assertFalse(b.consider(dict(psnr=40.5,ssim=0.985),3))
        self.assertTrue(b.consider(dict(psnr=42.,ssim=0.99),4))
        self.assertFalse(b.consider(dict(psnr=42.,ssim=0.99),5))
        other=BestTracker();other.load_state_dict(b.state_dict())
        self.assertEqual(other.state_dict(),b.state_dict())
        with self.assertRaises(ValueError): b.consider(dict(psnr=float('nan'),ssim=0.999),6)
        with self.assertRaises(ValueError): other.load_state_dict({'minima':{},'selected':None})

    @unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
    def test_amp_overflow_does_not_advance(self):
        setup(); e=Engine(4,1)
        ds=InfraredSRDataset(seed=config.SEED);ds.paths=ds.paths[:4]
        batch=next(iter(epoch_loader(ds,0)))
        calls=[0]
        def overflow_once(grad):
            calls[0]+=1
            return grad*float('inf') if calls[0]==1 else grad
        hook=next(e.model.parameters()).register_hook(overflow_once)
        try:e.update([batch])
        finally:hook.remove()
        self.assertGreaterEqual(e.skipped,1)
        self.assertEqual(e.updates,1);self.assertEqual(e.ema.num_updates,1)
        self.assertEqual(e.scheduler.last_update,1)

    @unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
    def test_exact_resume_mid_epoch_and_epoch_boundary(self):
        setup()
        ds=InfraredSRDataset(seed=config.SEED);ds.paths=ds.paths[:6]
        contract={'test':'mid_epoch'}
        e=Engine(4,1)
        batches=list(epoch_loader(ds,0))
        e.update([batches[0]])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'resume.pt'
            e.save(path,{'epoch':0,'offset':4},contract)
            losses=[]
            losses.append(e.update([batches[1]])['loss'])
            for batch in epoch_loader(ds,1):losses.append(e.update([batch])['loss'])
            final_model=copy.deepcopy(e.model.state_dict())
            final_ema=copy.deepcopy(e.ema.state_dict())
            final_optimizer=copy.deepcopy(e.optimizer.state_dict())
            final_rng=torch.get_rng_state().clone()
            del e
            restored=Engine(4,1);position=restored.restore(path,contract)
            self.assertEqual(position,{'epoch':0,'offset':4})
            after=[]
            for batch in epoch_loader(ds,0,4):after.append(restored.update([batch])['loss'])
            for batch in epoch_loader(ds,1):after.append(restored.update([batch])['loss'])
            self.assertEqual(after,losses)
            for name,value in final_model.items():torch.testing.assert_close(value,restored.model.state_dict()[name],rtol=0,atol=0)
            for name,value in final_ema['model'].items():torch.testing.assert_close(value,restored.ema.module.state_dict()[name],rtol=0,atol=0)
            for pid,state in final_optimizer['state'].items():
                for name,value in state.items():torch.testing.assert_close(value,restored.optimizer.state_dict()['state'][pid][name],rtol=0,atol=0)
            torch.testing.assert_close(final_rng,torch.get_rng_state(),rtol=0,atol=0)
            self.assertEqual(restored.updates,4);self.assertEqual(restored.ema.num_updates,4)
            with self.assertRaises(ValueError):restored.restore(path,{'test':'changed'})
            print('RESUME_EXACT_PASS full V1 CUDA FP16, partial batch and epoch boundary')


if __name__=='__main__':unittest.main(verbosity=2)
