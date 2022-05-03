"""Script that contains the LSTM networks."""
import torch


class LstmNet(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim1=48, hidden_dim2=32, hidden_dim3=16):
        super(LstmNet, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim1, dropout=0.2, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(2 * hidden_dim1, hidden_dim2, dropout=0.2, batch_first=True, bidirectional=True)
        self.lstm3 = torch.nn.LSTM(2 * hidden_dim2, hidden_dim3, dropout=0.2, batch_first=True, bidirectional=True)

    def forward(self, sequence):
        x, _ = self.lstm(sequence)
        x = torch.nn.ReLU()(x)
        x, _ = self.lstm2(x)
        x = torch.nn.ReLU()(x)
        _, (h, _) = self.lstm3(x)
        return h[-1, :, :]


class SiameseLSTM(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(SiameseLSTM, self).__init__()
        self.net = LstmNet(embedding_dim)

    def forward(self, sequence1, sequence2):
        output1 = self.net(sequence1)
        output2 = self.net(sequence2)
        return output1, output2
