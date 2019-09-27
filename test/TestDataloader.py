import unittest
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data.dataloader as dataloader
from Data import CaptionDataset


def cycle(data_loader):
    for batch in data_loader:
        yield batch


class MyTestCase(unittest.TestCase):
    def test_dataloader(self):
        flickr_dataset = CaptionDataset(
            os.path.join(os.path.abspath(os.path.join(__file__, "../..")), 'dataset', 'TRAIN.hdf5'))
        train_loader = dataloader.DataLoader(flickr_dataset, batch_size=3)
        train_loader = cycle(train_loader)
        data_batch = next(train_loader)

        self.assertEqual(data_batch['image'].shape, torch.Size([3, 3, 224, 224]))
        self.assertEqual(data_batch['caption'].shape, torch.Size([3, 38]))


if __name__ == '__main__':
    unittest.main()
