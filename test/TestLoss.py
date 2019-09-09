import unittest
from models import *
import torch
import torch.nn.functional as F


class MyTestCase(unittest.TestCase):
    def test_loss(self):
        vocab = {'SOS': 0, 'EOS': 1, 'start': 2, 'do': 3, 'relax': 4, 'cap': 5, 'PAD': 6}
        model = ShowTellModel(vocab=vocab, embedd_size=300, attention_size=512, hidden_size=512,
                              state_machine=None)
        test_input = torch.rand(2, 3, 224, 224)
        test_captions = torch.tensor([[2, 3, 1, 5, 6], [3, 1, 5, 6, 6]])
        _, test_loss, test_logits = model(test_input, test_captions)
        loss = 0
        for j in range(test_captions.shape[-1]):
            for i in range(2):
                if test_captions[i, j] == 6:
                    continue
                logit = test_logits[j][i:i + 1, :]
                print(logit.shape)
                print(test_captions[i, j])
                loss += F.cross_entropy(logit, test_captions[i:i + 1, j])
        # for i in range(test_captions.shape[-1] - 1):
        #     logit = test_logits[i]
        #     loss += F.cross_entropy(logit, test_captions[:, i])
        self.assertEqual(test_loss.item(), loss.item())


if __name__ == '__main__':
    unittest.main()
