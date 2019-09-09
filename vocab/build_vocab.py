import numpy as np
import json


def build():
    with open('glove.6B.100d.txt', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {'PAD': 0, 'EOS': 1, 'UNK': 2}
        embedding = np.zeros([len(lines) + 3, 100])
        for idx, line in enumerate(lines):
            line = line.strip('\n').split(' ')
            word = line[0]
            vector = np.array(line[1:]).astype(np.float)
            vocab[word] = idx + 3
            embedding[idx, :] = vector
    return vocab, embedding


if __name__ == '__main__':
    vocab, embedding = build()
    with open('vocab.json', 'w') as j:
        json.dump(vocab, j)
    np.save('embedding', embedding)

