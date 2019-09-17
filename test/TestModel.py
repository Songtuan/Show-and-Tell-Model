import unittest
from models import *
import torch

class MyTestCase(unittest.TestCase):
    def test_model(self):
        model = ShowTellModel(vocab={'start': 0, 'PAD': 1}, embed_dim=100)
        model.cuda()
        for name, parm in model.named_parameters():
            print(name)
            assert parm.is_cuda
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
