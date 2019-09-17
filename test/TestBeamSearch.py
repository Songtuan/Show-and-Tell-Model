import unittest
import os
import torchvision.transforms as trn
import json
from StateMachine import *
from BeamStateMachine import *
from skimage.io import imread
from skimage.transform import resize
from utils import util
from models import *
import torch
import torch.nn as nn

dir_main = os.path.abspath(os.path.join(__file__, "../.."))

class MyTestCase(unittest.TestCase):
    def test_beam_search(self):
        vocab_path = os.path.join(dir_main, 'vocab', 'vocab_pretrained.json')
        with open(vocab_path) as j:
            vocab = json.load(j)

        decoder_path = os.path.join(dir_main, 'decoder.pth')
        model = ShowTellModel(vocab=vocab, embed_dim=512)
        model.load_decoder(decoder_path)
        model.double()
        model.cuda()

        wordnet_id = 'n01321579'
        phases = util.get_hypernyms(wordnet_id=wordnet_id)
        state_machine, state_idx_mapping = util.build_state_machine(phases=phases, vocab=vocab)
        state_machine.add_state_idx_mapping(state_idx_mapping=state_idx_mapping)
        model.load_state_machine(state_machine=state_machine)

        img = imread(os.path.join(dir_main, 'img_15.jpg'))
        img = resize(img, (256, 256, 3))
        img = trn.ToTensor()(img)
        img = img.unsqueeze(dim=0)
        img = img.cuda()

        model.eval()
        seq, _ = model(img)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
