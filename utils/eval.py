import torch
import os
import argparse
import json
import numpy as np

from utils import util
from models import ShowTellModel
from Data import CaptionDataset
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm


dir_main = os.path.abspath(os.path.join(__file__, "../.."))
embed_dim = 512  # default embedding size

decoder_path = os.path.join(dir_main, 'decoder.pth')
encoder_path = os.path.join(dir_main, 'encoder_complete.pth')

parser = argparse.ArgumentParser("Evaluation an Show-and-Tell Model")
parser.add_argument('--folder', default='images', help='the folder of images')


if __name__ == '__main__':
    vocab_path = os.path.join(dir_main, 'vocab', 'vocab_pretrained.json')
    with open(vocab_path) as j:
        vocab = json.load(j)

    model = ShowTellModel(vocab=vocab, embed_dim=embed_dim)
    model.load_decoder(PATH=decoder_path)
    model.load_encoder(PATH=encoder_path)
    model.double()
    model.cuda()

    # create eval data-loader
    eval_set_path = os.path.join(dir_main, 'dataset', 'VAL.hdf5')
    eval_set = CaptionDataset(input_file=eval_set_path)
    eval_loader = DataLoader(dataset=eval_set, batch_size=100)

    preds_caps = []
    real_caps = []
    bleu_4 = []

    for idx, batch in enumerate(tqdm(eval_loader)):
        # iterate through each batch
        # make sure model is in eval mode
        model.eval()

        # load the images' data and captions in each batch
        imgs = batch['image']
        imgs = imgs.cuda()
        caps = batch['caption']
        caps = caps.cuda()

        with torch.no_grad():
            preds, _ = model(imgs)

        assert preds.shape[-1] == caps.shape[0]

        for t in range(preds.shape[-1]):
            pred_cap = preds[:, t].cpu().numpy().tolist()
            pred_cap = util.decode_str(vocab=vocab, cap=pred_cap)
            preds_caps.append(pred_cap)

            real_cap = caps[t, :].cpu().numpy().tolist()
            real_cap = util.decode_str(vocab=vocab, cap=real_cap)
            real_caps.append(real_cap)

            bleu_4_single = sentence_bleu(references=real_cap, hypothesis=pred_cap)
            bleu_4.append(bleu_4_single)

        print(preds_caps[idx * 100])
        print('*********')
        print(real_caps[idx * 100])


    # bleu_4 = corpus_bleu(list_of_references=real_caps, hypotheses=preds_caps)
    bleu_4 = np.mean(np.asarray(bleu_4))
    print(bleu_4)









