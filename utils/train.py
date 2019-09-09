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


parser = argparse.ArgumentParser("Train an Show-and-Tell Model on Flickr8K")
parser.add_argument('--epochs', default=150, help='setting the number of training epochs')
parser.add_argument('--learning_rate', default=4e-4, help='setting the learning rate')
parser.add_argument('--batch_size', default=10, help='setting the batch size')
parser.add_argument('--beam_size', default=3, help='setting the beam size for beam search')
parser.add_argument('--attention_size',required=False, help='setting attention size')
parser.add_argument('--hidden_size', required=False, help='setting hidden size')

def cycle(data_loader):
    for batch in data_loader:
        yield batch


if __name__ == '__main__':
    dir_main = os.path.abspath(os.path.join(__file__, "../.."))  # the root directory of project

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

    # create optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        # training loop
        # ensure model is in training mode
        model.train()

        for data_batch in training_loader:
            # load the batch data
            imgs, caps = data_batch['image'], data_batch['caption']
            # perform back-propagation over batch
            _, loss = model(imgs, caps)
            loss.backward()
            optimizer.step()


