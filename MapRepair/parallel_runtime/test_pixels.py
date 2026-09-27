import os,unittest
from unittest.mock import patch
import numpy as np
from .pixels import cpu_composite,composite_bgra

class PixelTests(unittest.TestCase):
    def test_legacy_parity_all_foreground_alpha(self):
        a,v=np.mgrid[:256,:256]
        image=np.stack([v,255-v,v,a],-1).astype(np.uint8)
        for bg in ([0,0,0],[255,255,255],[17,128,241]):
            alpha=image[:,:,3:4].astype(np.float32)/255
            legacy=(image[:,:,:3]*alpha+np.array(bg,np.float32)*(1-alpha)+.5).astype(np.uint8)
            np.testing.assert_array_equal(cpu_composite(image,bg),legacy)
            np.testing.assert_array_equal(composite_bgra(image,bg,'cpu'),legacy)
    def test_invalid_input_and_budget(self):
        image=np.zeros((2,3,4),np.uint8)
        for background in ([256,0,0],[1.5,0,0],[0,0,float('nan')]):
            with self.assertRaises(ValueError):composite_bgra(image,background,'cpu')
        with patch('parallel_runtime.pixels.memory_budget',return_value=1):
            with self.assertRaises(MemoryError):composite_bgra(image,backend='cpu')
    def test_auto_falls_back_after_device_failure(self):
        image=np.full((256,256,4),80,np.uint8)
        with patch('parallel_runtime.pixels.certified_device',return_value='cuda'),patch('parallel_runtime.pixels.torch_composite',side_effect=RuntimeError('device lost')):
            np.testing.assert_array_equal(composite_bgra(image,backend='auto'),cpu_composite(image,[255]*3))
            with self.assertRaises(RuntimeError):composite_bgra(image,backend='hip')
if __name__=='__main__':unittest.main()
