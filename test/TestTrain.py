import unittest
import torch
import os
import torch.nn as nn
import argparse
import numpy as np
import json

from torch.utils.data import DataLoader
from torch import optim
from models.UpDownModel import UpDownCaptioner
from allennlp.training.metrics import BLEU



class MyTestCase(unittest.TestCase):
    def test_training(self):
        dir_main = os.path.abspath(os.path.join(__file__, "../.."))  # the root directory of project
        # embedding_path = os.path.join(dir_main, 'vocab', 'embedding.npy')
        # vocab_path = os.path.join(dir_main, 'vocab', 'vocab.json')
        # embeddings = np.load(embedding_path)
        # embeddings = torch.from_numpy(embeddings)
        # with open(vocab_path) as j:
        #     vocab = json.load(j)
        vocab = {'<start>': 0, '<pad>': 1, '<end>': 2, 'to': 3, 'do': 4, 'it': 5, 'go': 6, 'as': 7}
        model = UpDownCaptioner(vocab=vocab, embed_dim=10)
        # model.double()
        model.cuda()

        test_input = torch.rand(1, 3, 224, 224).cuda()
        # test_input = [test_input[i, :, :, :] for i in range(test_input.shape[0])]
        caption = torch.tensor([[0, 5, 6, 3, 6, 2]]).long().cuda()
        # optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9, weight_decay=0.001)
        # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iteration: 1 - iteration / 500)
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=4e-4, weight_decay=0.001)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iteration: 1 - iteration / 500)

        for epoch in range(500):
            model.train()
            output_dict = model(test_input, caption)
            loss = output_dict['loss']
            model.zero_grad()

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), True)

            optimizer.step()
            lr_scheduler.step()

            model.eval()
            bleu_eval = BLEU(exclude_indices={0, 1, 2})
            output_dict = model(test_input)
            seq = output_dict['seq']
            bleu_eval(predictions=seq, gold_targets=caption)
            bleu = bleu_eval.get_metric()['BLEU']
            if epoch % 10 == 0:
                print(loss)
                print(bleu)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
