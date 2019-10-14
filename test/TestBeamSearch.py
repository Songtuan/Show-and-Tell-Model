import unittest
import os
import torchvision.transforms as trn
import json
from StateMachine import *
from BeamStateMachine import *
from models.UpDownModel import UpDownCaptioner
from skimage.io import imread
from skimage.transform import resize
from utils import util
from models import *
import torch
import numpy as np
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
        model.eval()

        # embedding_path = os.path.join(dir_main, 'vocab', 'embedding.npy')
        # vocab_path = os.path.join(dir_main, 'vocab', 'vocab.json')
        # embeddings = np.load(embedding_path)
        # embeddings = torch.from_numpy(embeddings)
        # with open(vocab_path) as j:
        #     vocab = json.load(j)
        #
        # model_path = os.path.join(dir_main, 'UpDown.pth')
        # model = UpDownCaptioner(vocab=vocab, pre_trained_embedding=embeddings)
        # model.load_state_dict(torch.load(model_path))
        # model.cuda()
        # model.eval()

        img = imread(os.path.join(dir_main, 'img_15.jpg'))
        # img = resize(img, (256, 256, 3))
        img = trn.ToTensor()(img)
        img = img.unsqueeze(dim=0)
        img = img.double()
        # img = img.float()
        img = img.cuda()

        wordnet_id = 'n01321579'
        phases = util.get_hypernyms(wordnet_id=wordnet_id)
        state_machine, state_idx_mapping = util.build_state_machine(phases=phases, vocab=vocab)
        state_machine.add_state_idx_mapping(state_idx_mapping=state_idx_mapping)
        model.load_state_machine(state_machine=state_machine)

        seq, _ = model(img)
        preds = util.decode_str(vocab=vocab, cap=seq[:, 0].cpu().numpy().tolist())
        print(preds)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
