import unittest
from modules import Resnet
import torch

class MyTestCase(unittest.TestCase):
    def test_shape(self):
        encoder = Resnet(encoded_image_size=14)
        test_img = torch.rand(10, 3, 224, 224)
        output = encoder(test_img)
        self.assertEqual(output.shape, torch.Size([10, 2048, 14, 14]))


if __name__ == '__main__':
    unittest.main()
