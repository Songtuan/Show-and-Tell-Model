import unittest
import torch

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg


class MyTestCase(unittest.TestCase):
    def test_faster_rcnn(self):
        cfg.MODEL.ROI_BOX_HEAD.RETURN_FC_FEATS = True
        model = build_detection_model(cfg)
        model.eval()

        test_input = torch.rand(3, 255, 255)
        test_input = [test_input]
        feats, _ = model(test_input)
        print(feats.shape)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
