import unittest
import os
import matplotlib.pyplot as plt
import numpy as np
from Data import Flickr8kDataset


class MyTestCase(unittest.TestCase):
    def test_dataset(self):
        flickr_dataset = Flickr8kDataset(os.path.join(os.path.abspath(os.path.join(__file__ ,"../..")), 'dataset', 'TRAIN.hdf5'))
        fig = plt.figure()
        for i in range(6):
            data = flickr_dataset[i]
            print(data['caption'])
            print(data['caption_unencode'])
            ax = plt.subplot(1, 6, i + 1)
            img = data['image'].permute(1, 2, 0).numpy()
            plt.imshow(img)
        plt.show()

        flickr_dataset = Flickr8kDataset(
            os.path.join(os.path.abspath(os.path.join(__file__, "../..")), 'dataset', 'VAL.hdf5'))
        fig = plt.figure()
        for i in range(6):
            data = flickr_dataset[i]
            print(data['caption'])
            print(data['caption_unencode'])
            ax = plt.subplot(1, 6, i + 1)
            img = data['image'].permute(1, 2, 0).numpy()
            plt.imshow(img)
        plt.show()

        flickr_dataset = Flickr8kDataset(
            os.path.join(os.path.abspath(os.path.join(__file__, "../..")), 'dataset', 'TEST.hdf5'))
        fig = plt.figure()
        for i in range(6):
            data = flickr_dataset[i]
            print(data['caption'])
            print(data['caption_unencode'])
            ax = plt.subplot(1, 6, i + 1)
            img = data['image'].permute(1, 2, 0).numpy()
            plt.imshow(img)
        plt.show()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
