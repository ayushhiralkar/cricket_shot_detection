import torch
import torch.nn as nn

class CricketMLP(nn.Module):
    """
    Multi-layer Perceptron for frame-level cricket shot pose classification.
    Input: 17 feature dimensions (joint angles, velocities, distances).
    Output: 8 classes (cricket shots).
    """
    def __init__(self, input_dim=17, num_classes=8, hidden_dims=[64, 32]):
        super(CricketMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class CricketLSTM(nn.Module):
    """
    LSTM (Long Short-Term Memory) Network for sequence-level cricket shot classification.
    Input sequence shape: [batch_size, sequence_length, feature_dim]
    Output: logits for 8 classes.
    """
    def __init__(self, input_dim=17, num_classes=8, hidden_dim=64, num_layers=2):
        super(CricketLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        # Classifier head operating on the final hidden state
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Out shape: [batch_size, seq_len, hidden_dim]
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
