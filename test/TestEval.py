import unittest
import torch
import os
import argparse
import numpy as np
import json

from torch.utils.data import DataLoader
from Data import CaptionDataset
from models.UpDownModel import UpDownCaptioner
from allennlp.training.metrics import BLEU
from tqdm import tqdm


class MyTestCase(unittest.TestCase):
    def test_eval(self):
        dir_main = os.path.abspath(os.path.join(__file__, "../.."))

        eval_set_path = os.path.join(dir_main, 'dataset', 'VAL.hdf5')
        eval_set = CaptionDataset(eval_set_path)
        eval_loader = DataLoader(dataset=eval_set, batch_size=10)

        embedding_path = os.path.join(dir_main, 'vocab', 'embedding.npy')
        vocab_path = os.path.join(dir_main, 'vocab', 'vocab.json')
        embeddings = np.load(embedding_path)
        embeddings = torch.from_numpy(embeddings)
        with open(vocab_path) as j:
            vocab = json.load(j)

        model_path = os.path.join(dir_main, 'UpDown.pth')
        model = UpDownCaptioner(vocab=vocab, pre_trained_embedding=embeddings)
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()

        bleu_eval = BLEU(exclude_indices={vocab['<start>'], vocab['<end>'], vocab['<pad>']},
                         ngram_weights=[0.5, 0.5, 0, 0])

        with torch.no_grad():
            for data_batch in tqdm(eval_loader):
                # load the batch data
                imgs, caps = data_batch['image'], data_batch['caption']
                imgs = imgs.cuda()

                output_dict = model(imgs)
                seq = output_dict['seq']
                bleu_eval(predictions=seq, gold_targets=caps)
        bleu_score = bleu_eval.get_metric()['BLEU']
        print(bleu_score)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
