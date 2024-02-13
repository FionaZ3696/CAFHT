import torch
import torch.nn as nn


    
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #out, _ = self.lstm(x, (h0, c0))
        #_, (h_n, c_n) = self.lstm(x.unsqueeze(-1))
        out, _ = self.lstm(x.unsqueeze(-1))
        out = self.relu(out)
        out = self.fc(out)
        out = out.squeeze(-1)
        return out



class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #out, _ = self.lstm(x, (h0, c0))
        #_, (h_n, c_n) = self.lstm(x.unsqueeze(-1))
        out, _ = self.lstm(x)
        #out = self.relu(out)
        out = self.fc(out)
        return out



