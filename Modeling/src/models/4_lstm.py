

import torch
import torch.nn as nn

class SineLSTM(nn.Module):
    """
    A simple two-layer LSTM model for univariate time-series prediction.
    """
    def __init__(self, input_size=1, hidden_size=64):
        super(SineLSTM, self).__init__()
        self.hidden_size = hidden_size
        # First LSTM cell: from input to hidden
        self.lstm1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        # Second LSTM cell: from hidden to hidden
        self.lstm2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        # Final linear layer: from hidden to scalar output
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, future=0):
        """
        Forward pass through the model.
        Args:
            x: Tensor of shape (batch_size, seq_len), the input time-series.
            future: int, number of future time-steps to predict beyond input.
        Returns:
            outputs: Tensor of shape (batch_size, seq_len + future)
        """
        outputs = []
        batch_size = x.size(0)
        
        # Initialize hidden and cell states for both layers (on same device as input)
        h1 = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=x.device)
        c1 = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=x.device)
        h2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=x.device)
        c2 = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=x.device)
        
        # Iterate over each time step in the input sequence
        for t in x.split(1, dim=1):
            inp = t.squeeze(1)  # shape: (batch_size,)
            h1, c1 = self.lstm1(inp.unsqueeze(1), (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            out = self.linear(h2)  # shape: (batch_size, 1)
            outputs.append(out)
        
        # Predict future time steps, feeding back last output
        for _ in range(future):
            h1, c1 = self.lstm1(out, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            out = self.linear(h2)
            outputs.append(out)
        
        # Concatenate all outputs along time dimension
        return torch.cat(outputs, dim=1)