import unittest
import os
import numpy as np
import torch
import json


class MyTestCase(unittest.TestCase):
    def test_load_embedding(self):
        dir_main = os.path.abspath(os.path.join(__file__, "../.."))
        embedding_path = os.path.join(dir_main, 'vocab', 'embedding.npy')
        vocab_path = os.path.join(dir_main, 'vocab', 'vocab.json')

        embeddings = np.load(embedding_path)
        embeddings = torch.from_numpy(embeddings)

        with open(vocab_path) as j:
            vocab = json.load(j)
        print(embeddings.shape)
        self.assertEqual(len(vocab), embeddings.shape[0])


if __name__ == '__main__':
    unittest.main()
