import torch
import os
import argparse
import json
import torchvision.transforms as trn
import matplotlib.pyplot as plt
import modules.BeamSearch as BeamSearch

from skimage.io import imread
from skimage.transform import resize
from utils import util
from models import ShowTellModel

dir_main = os.path.abspath(os.path.join(__file__, "../.."))

# load vocabulary
vocab_path = os.path.join(dir_main, 'vocab', 'vocab_pretrained.json')
with open(vocab_path) as j:
    vocab = json.load(j)

# load model
decoder_path = os.path.join(dir_main, 'decoder.pth')
encoder_path = os.path.join(dir_main, 'encoder_complete.pth')
model = ShowTellModel(vocab=vocab, embed_dim=512)
model.load_decoder(decoder_path)
# model.load_encoder(encoder_path)
model.double()
model.cuda()


if __name__ == '__main__':
    model.eval()
    folder_name = 'n01321579'
    wordnet_id = folder_name
    phases = util.get_hypernyms(wordnet_id=wordnet_id)
    trigger_words = ''
    for phrase in phases:
        trigger_words += phrase + ','
    print(phases)
    print('***************')
    state_machine, state_idx_mapping = util.build_state_machine(phases=phases, vocab=vocab)
    state_machine.add_state_idx_mapping(state_idx_mapping=state_idx_mapping)
    os.mkdir(os.path.join(dir_main, folder_name + '_img'))
    folder_path = folder_name + '_img'
    # model.load_state_machine(state_machine=state_machine)
    for idx, file_name in enumerate(os.listdir(os.path.join(dir_main, folder_name))):
        print(file_name)
        if idx % 2 == 0:
            num_figures = min(2, len(os.listdir(os.path.join(dir_main, folder_name))) - idx)
            fig, axes = plt.subplots(nrows=num_figures, ncols=2, figsize=(12, 12))
            fig.suptitle(trigger_words)
            axes = axes.flat
            count = 0
        images = imread(os.path.join(dir_main, folder_name, file_name))
        img = resize(images, (256, 256, 3))
        img = trn.ToTensor()(img)
        # normalize = trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # img = normalize(img)
        img = img.unsqueeze(dim=0)
        img = img.double()
        img = img.cuda()
        # print(img)

        model.load_state_machine(state_machine=state_machine)
        seq, _ = model(img)
        preds = util.decode_str(vocab=vocab, cap=seq[:, 0].cpu().numpy().tolist())
        cap = ''
        for word in preds:
            cap += word + ' '
        print(cap)
        ax = next(axes)
        ax.set_title(cap)
        ax.imshow(images)

        model.load_state_machine(state_machine=BeamSearch.BeamSearch.build_default_state_machine(vocab=vocab))
        seq, _ = model(img)
        preds = util.decode_str(vocab=vocab, cap=seq[:, 0].cpu().numpy().tolist())
        cap = ''
        for word in preds:
            cap += word + ' '
        print(cap)
        ax = next(axes)
        ax.set_title(cap)
        ax.imshow(images)
        if count == 1:
            plt.tight_layout()
            fig.savefig(os.path.join(dir_main, folder_path, 'img_{}.png'.format(idx)))
        count += 1
        # print(preds)
        # print('**********************')
    # plt.subplots_adjust(wspace=5, hspace=10)
    # plt.tight_layout()
    # plt.show()

