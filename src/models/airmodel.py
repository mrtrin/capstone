import torch.nn as nn

class AirModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50, num_layers=1, batch_first=True):
        super().__init__()
        print(f'Creating model with input {input_size}, output_size {output_size}')
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
