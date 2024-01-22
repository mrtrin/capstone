import torch.nn as nn


class AirModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, output_size)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
