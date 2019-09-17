import unittest
from modules import DecoderAttCell
import torch


class MyTestCase(unittest.TestCase):
    def test_shape(self):
        decoder_cell = DecoderAttCell(encoder_dim=2048, attention_dim=512, embed_dim=512,
                                      voocab_size=3000, decoder_dim=512)
        img = torch.rand(5, 2048, 14, 14)
        output, _ = decoder_cell(torch.tensor([50, 100, 30, 70, 6]), img)
        self.assertEqual(output.shape, torch.Size([5, 3000]))


if __name__ == '__main__':
    unittest.main()
