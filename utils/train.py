import torch
import os
import argparse
import numpy as np
import json

from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from models import ShowTellModel
from Data import Flickr8kDataset
from nltk.translate.bleu_score import corpus_bleu


parser = argparse.ArgumentParser("Train an Show-and-Tell Model on Flickr8K")
parser.add_argument('--epochs', default=150, help='setting the number of training epochs')
parser.add_argument('--learning_rate', default=4e-4, help='setting the learning rate')
parser.add_argument('--batch_size', default=10, help='setting the batch size')
parser.add_argument('--beam_size', default=3, help='setting the beam size for beam search')
parser.add_argument('--attention_size',required=False, help='setting attention size')
parser.add_argument('--hidden_size', required=False, help='setting hidden size')


def decode_str(id_to_word, cap):
    '''
    map a caption to words
    :param id_to_word: word_id -> word, Dict[id] -> word
    :param cap: caption, list
    :return: list
    '''
    caption = []
    for token_id in cap:
        if token_id not in [0, 1, 2]:
            caption.append(id_to_word[token_id])
    return caption


if __name__ == '__main__':
    dir_main = os.path.abspath(os.path.join(__file__, "../.."))  # the root directory of project
    PATH = os.path.join(dir_main, 'Show-and-Tell.pth')

    # load cmd parameters
    opt = parser.parse_args()
    epochs = opt.epochs
    lr = opt.learning_rate
    batch_size = opt.batch_size
    beam_size = opt.beam_size

    # load the pre-trained word embedding and vocab
    embedding_path = os.path.join(dir_main, 'vocab', 'embedding.npy')
    vocab_path = os.path.join(dir_main, 'vocab', 'vocab.json')
    embeddings = np.load(embedding_path)
    embeddings = torch.from_numpy(embeddings)
    with open(vocab_path) as j:
        vocab = json.load(j)
    id_to_word = {idx: word for idx, word in enumerate(vocab.keys())}

    # load training set
    training_set_path = os.path.join(dir_main, 'dataset', 'TRAIN.hdf5')
    training_set = Flickr8kDataset(training_set_path)

    # load eval set
    eval_set_path = os.path.join(dir_main, 'dataset', 'VAL.hdf5')
    eval_set = Flickr8kDataset(eval_set_path)

    # build data-loaders for both training set and eval set
    # make both of them iterable
    training_loader = DataLoader(dataset=training_set, batch_size=batch_size)
    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size)

    # load model
    model = ShowTellModel(vocab=vocab, pretrained_embedding=embeddings)
    model.double()
    model.cuda()

    # create optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # if the current bleu_4 score better than
    # the best produced before, store the model
    # and record the bleu_4 score
    best_bleu_4 = 0

    for epoch in tqdm(range(epochs)):
        # training loop
        # ensure model is in training mode
        model.train()

        for data_batch in training_loader:
            # load the batch data
            imgs, caps = data_batch['image'], data_batch['caption']
            imgs = imgs.cuda()
            caps = caps.cuda()
            # perform back-propagation over batch
            _, loss = model(imgs, caps)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            # evaluation every 10 epochs
            model.eval()
            preds_caps = []
            real_caps = []

            for data_batch in eval_loader:
                # load the batch data
                imgs, caps = data_batch['image'], data_batch['caption']
                # get the prediction
                preds, _ = model(imgs)

                assert preds.shape[-1] == caps.shape[0]

                for t in range(preds.shape[-1]):
                    preds_caps.append(decode_str(id_to_word=id_to_word, cap=preds[:, t].numpy().tolist()))
                    real_caps.append(decode_str(id_to_word=id_to_word, cap=caps[t, :].numpy().tolist()))

            bleu_4 = corpus_bleu(list_of_references=real_caps, hypotheses=preds_caps)
            print(preds_caps[epoch])
            print('Epoch: {}, loss: {}, bleu_4: {}'.format(epoch, loss, bleu_4))
            if bleu_4 > best_bleu_4:
                # if the current bleu_4 score better than
                # the best produced before, store the model
                # and record the bleu_4 score
                best_bleu_4 = bleu_4
                torch.save(model.state_dict(), PATH)





