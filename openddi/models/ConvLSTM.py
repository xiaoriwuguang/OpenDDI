import torch
import torch.nn as nn

class ConvLSTM(nn.Module):
    """
    ConvLSTM model combining convolutional and LSTM layers for sequence processing.
    
    This model uses 1D convolutional layers followed by LSTM layers to process
    sequential data, with global pooling and fully connected layers for classification.
    """
    
    def __init__(self, feature: int, hidden1: int, hidden2: int,
                 num_relations: int, num_classes: int, dropout: float = 0.2, timesteps=1):
        """
        Initialize ConvLSTM model.
        
        Args:
            feature: Input feature dimension
            hidden1: First hidden layer dimension for conv and LSTM
            hidden2: Second hidden layer dimension for conv and LSTM
            num_relations: Number of relation types (unused in current implementation)
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
            timesteps: Number of timesteps for sequence processing
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature = int(feature)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.hidden3 = int(hidden2 / 2)
        self.dropout_ratio = dropout

        # First convolution block
        self.liner1 = nn.Linear(self.feature * 2, int(self.feature))

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.hidden1, kernel_size=32, stride=32, padding=2)
        self.bn1 = nn.BatchNorm1d(self.hidden1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        
        # Second convolution block
        self.conv2 = nn.Conv1d(in_channels=self.hidden1, out_channels=self.hidden2, kernel_size=4, stride=4, padding=1)
        self.bn2 = nn.BatchNorm1d(self.hidden2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        
        # Global pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # Global max pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=self.hidden2, hidden_size=self.hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(self.dropout_ratio)
        self.lstm2 = nn.LSTM(input_size=self.hidden1, hidden_size=self.hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(self.dropout_ratio)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden2 * 2, self.num_classes)
        
    def forward(self, data_o, idx):
        """
        Forward pass of the ConvLSTM model.
        
        Args:
            data_o: Graph data object containing node features, edge indices, and edge types
            idx: Batch indices containing pairs of nodes to process
            
        Returns:
            Output logits for the given node pairs
        """
        x_o, edge_index, e_type = data_o.x, data_o.edge_index, data_o.edge_type

        a_idx = torch.as_tensor(list(idx[0]), dtype=torch.long, device=x_o.device)
        b_idx = torch.as_tensor(list(idx[1]), dtype=torch.long, device=x_o.device)

        # Alternative feature concatenation approach (commented out)
        # xa = x_o[a_idx]
        # xb = x_o[b_idx]
        # x = torch.cat([xa, xb], dim=1)  # Concatenate features of a and b
        # x = self.liner1(x)  
        
        x = x_o.unsqueeze(1)
        # First convolution block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Second convolution block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        
        # LSTM section
        x = x.permute(0, 2, 1)  
        lstm_out, _ = self.lstm1(x)  
        lstm_out = self.dropout1(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)  
        lstm_out = self.dropout2(lstm_out)  
        lstm_out = lstm_out[:, -1, :]  # Take the last timestep output
        
        
        # Output layer
        out1 = lstm_out[a_idx]
        out2 = lstm_out[b_idx]
        lstm_out = torch.cat([out1, out2], dim=1)
        out = self.fc(lstm_out)
        
        return out