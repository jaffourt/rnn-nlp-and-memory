import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    """

    Args:
        None

    Inputs:
        None


    """

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm_size = hidden_size
        self.embedding = nn.Embedding(output_size, input_size)
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x, prev_state):
        input = self.embedding(x)
        output, state = self.lstm(input, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


class RNN(nn.Module):
    """
    RNN for character based text generation

    Args:
        None

    Inputs:
        None


    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.5)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        hidden = self.h2h(hidden)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        # output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
