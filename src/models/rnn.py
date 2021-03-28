from src.constants import RNNParams, Globals
import torch


class RNN(torch.nn.Module):
    """
    Simple RNN model with the following structure:

    (input|hidden) -> linear -> hidden_out
    (input|hidden) -> linear -> LogSoftmax -> output
    """
    def __init__(self):
        super(RNN, self).__init__()

        self.input_size = Globals.N_LETTERS
        self.hidden_size = RNNParams.N_HIDDEN
        self.output_size = Globals.N_CATEGORIES

        self.hidden_linear = torch.nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.output_linear = torch.nn.Linear(self.input_size + self.hidden_size, self.output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        combined = torch.cat((input_tensor, hidden), 1)
        hidden = self.hidden_linear(combined)
        output = self.output_linear(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Initializes the hidden layer with zeros
        :return: initialized layer
        """
        return torch.zeros(1, self.hidden_size)