import torch
import os
import argparse
import numpy as np
import json

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from models.UpDownModel import UpDownCaptioner
from Data import CaptionDataset
from allennlp.training.metrics import BLEU


parser = argparse.ArgumentParser("Train an Show-and-Tell Model on Flickr8K")
parser.add_argument('--epochs', default=150, help='setting the number of training epochs')
parser.add_argument('--learning_rate', default=4e-4, help='setting the learning rate')
parser.add_argument('--batch_size', default=5, help='setting the batch size')
parser.add_argument('--beam_size', default=3, help='setting the beam size for beam search')
parser.add_argument('--attention_size',required=False, help='setting attention size')
parser.add_argument('--hidden_size', required=False, help='setting hidden size')

dir_main = os.path.abspath(os.path.join(__file__, "../.."))  # the root directory of project
model_store_path = os.path.join(dir_main, 'UpDown.pth')


if __name__ == '__main__':
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
    training_set = CaptionDataset(training_set_path)

    # load eval set
    eval_set_path = os.path.join(dir_main, 'dataset', 'VAL.hdf5')
    eval_set = CaptionDataset(eval_set_path)

    # build data-loaders for both training set and eval set
    # make both of them iterable
    training_loader = DataLoader(dataset=training_set, batch_size=batch_size)
    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size)

    # load model
    model = UpDownCaptioner(vocab=vocab, pre_trained_embedding=embeddings)
    model.cuda()  # put the model to gpu

    # create optimizer
    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # if the current bleu_4 score better than
    # the best produced before, store the model
    # and record the bleu_4 score
    best_bleu = 0

    for epoch in tqdm(range(epochs)):
        model.train()

        for data_batch in training_loader:
            imgs, caps = data_batch['image'], data_batch['caption']
            imgs = imgs.cuda()
            caps = caps.cuda()

            # perform back-propagation over batch
            output_dict = model(imgs, caps)
            loss = output_dict['loss']
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            # evaluation every 10 epochs
            model.eval()
            bleu_scores = []
            bleu_eval = BLEU(exclude_indices={vocab['<start>'], vocab['<end>'], vocab['<pad>']})
            with torch.no_grad():
                for data_batch in eval_loader:
                    # load the batch data
                    imgs, caps = data_batch['image'], data_batch['caption']
                    imgs = imgs.cuda()

                    output_dict = model(imgs)
                    seq = output_dict['seq']
                    bleu_eval(predictions=seq, gold_targets=caps)
                    bleu_score = bleu_eval.get_metric()
                    bleu_scores.append(bleu_score)
            final_bleu = np.mean(np.asarray(bleu_scores))
            if final_bleu > best_bleu:
                best_bleu = final_bleu
                torch.save(model.state_dict(), model_store_path)





