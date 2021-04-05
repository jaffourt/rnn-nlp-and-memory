import torch
import torch.nn as nn

from dataset import Dataset
from models import RNN
from argparse import Namespace

flags = Namespace(
    batch_size=32,
    embedding_size=65,
    max_batch=10000,
    hidden_size=256,
    gradients_norm=1.0,
    learning_rate=5e-5,
    epochs=1000
)

# TODO: nothing right now...
dataset = Dataset(max_batch=flags.max_batch, embedding_size=flags.embedding_size, word_level=False)

# TODO: n_vocab === train set
model = RNN(input_size=dataset.vocabulary.vocab_size, hidden_size=flags.hidden_size,
            output_size=dataset.vocabulary.vocab_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=flags.learning_rate)

# training parameters
epoch = 1
hidden = model.init_hidden()
total_loss = 0
iterator = 0
all_losses = []

for i in range(flags.epochs * flags.max_batch):
    inputs, targets = dataset()
    targets.unsqueeze_(-1)

    if iterator >= epoch * flags.max_batch:
        epoch += 1

    model.train()
    hidden = model.init_hidden()
    optimizer.zero_grad()

    loss = 0

    for j in range(inputs.size(0)):
        output, hidden = model(inputs[j], hidden)
        l = criterion(output, targets[j])
        loss += l

    loss.backward()

    total_loss += loss.item() / inputs.size(0)

    # for p in model.parameters():
    #     p.data.add_(p.grad.data, alpha=-flags.learning_rate)
    torch.nn.utils.clip_grad_norm_(model.parameters(), flags.gradients_norm)

    optimizer.step()

    iterator += 1
    if iterator % 100 == 0:
        print("Iteration: %d, Loss: %.4f" % (iterator, loss.item() / inputs.size(0)))
        all_losses.append(total_loss / 100)
        total_loss = 0

    if iterator % 1000 == 0:

        with torch.no_grad():  # no need to track history in sampling
            from random import choice
            from string import ascii_letters

            start_letter = choice(ascii_letters.lower())
            input = dataset.vocabulary.one_hot([dataset.vocabulary.item_to_index[start_letter]])
            output_name = start_letter

            for i in range(flags.embedding_size):
                output, hidden = model(input[0], hidden)
                topv, topi = output.topk(1)
                letter = dataset.vocabulary.index_to_item[int(topi[0][0])]
                output_name += letter
                input = dataset.vocabulary.one_hot([dataset.vocabulary.item_to_index[letter]])

        print(output_name)
