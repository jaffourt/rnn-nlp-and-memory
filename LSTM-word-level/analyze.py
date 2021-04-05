import torch
import numpy as np
from argparse import Namespace
from models import LSTM
from dataset import Dataset


def predict(device, model, vocabulary, top_k=5):
    model.eval()
    words = [vocabulary.index_to_word[np.random.randint(vocabulary.vocab_size)]]

    state_h, state_c = model.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocabulary.word_to_index[w]]]).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))

    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(vocabulary.index_to_word[choice])

    for _ in range(64):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(vocabulary.index_to_word[choice])

    print(' '.join(words).encode('utf-8'))


flags = Namespace(
    batch_size=16,
    embedding_size=64,
    max_batch=900,
    lstm_size=128,
    gradients_norm=5.0,
    epochs=50
)

dataset = Dataset(batch_size=flags.batch_size, embedding_size=flags.embedding_size, max_batch=flags.max_batch)
model = LSTM(n_vocab=dataset.vocabulary.vocab_size, embedding_size=flags.embedding_size, lstm_size=flags.lstm_size)

torch.load('states/model-20.pth')
