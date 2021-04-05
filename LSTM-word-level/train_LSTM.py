import torch
import torch.nn as nn
from argparse import Namespace

from dataset import Dataset
from models import LSTM
from analyze import predict

flags = Namespace(
    batch_size=32,
    embedding_size=16,
    max_batch=100,
    hidden_size=128,
    gradients_norm=1.0,
    learning_rate=1e-3,
    epochs=5
)

# TODO: nothing right now...
dataset = Dataset(batch_size=flags.batch_size, embedding_size=flags.embedding_size,
                  max_batch=flags.max_batch, batch_first=True)

# TODO: n_vocab === train set
model = LSTM(input_size=flags.embedding_size, hidden_size=flags.hidden_size, output_size=dataset.vocabulary.vocab_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# hidden state and cell state (needed for LSTM)
state_h, state_c = model.zero_state(flags.batch_size)
state_h = state_h.to(device)
state_c = state_c.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=flags.learning_rate)

epoch = 1
iterator = 0
for i in range(flags.epochs * flags.max_batch):

    if iterator >= epoch * flags.max_batch:
        epoch += 1
        state_h, state_c = model.zero_state(flags.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

    iterator += 1
    inputs, targets = dataset()

    model.train()
    optimizer.zero_grad()

    x = torch.tensor(inputs).to(device)
    y = torch.tensor(targets).to(device)

    logits, (state_h, state_c) = model(x, (state_h, state_c))

    loss = criterion(logits.transpose(1, 2), y)

    loss_value = loss.item()
    loss.backward()

    state_h = state_h.detach()
    state_c = state_c.detach()

    torch.nn.utils.clip_grad_norm_(model.parameters(), flags.gradients_norm)

    optimizer.step()

    if iterator % 100 == 0:
        print('Epoch: {}/{}'.format(epoch, flags.epochs),
              'Iteration: {}'.format(iterator),
              'Loss: {}'.format(loss_value))

    if iterator % (epoch*flags.max_batch) == 0:
        predict(device, model, dataset.vocabulary, top_k=5)
        torch.save(model.state_dict(),
                   'LSTM-word-level/states_testing/epoch-{}.pth'.format(epoch))
