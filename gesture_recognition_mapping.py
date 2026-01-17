import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Dataset class for sequences of sensor readings with labels (gestures)
class GestureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # shape: (num_samples, seq_len, features)
        self.labels = labels        # shape: (num_samples,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), self.labels[idx]

# LSTM Model for gesture classification
class GestureLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, layer_dim=2, output_dim=12, dropout=0.3):
        super(GestureLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: batch_size, seq_len, hidden_dim
        out = out[:, -1, :]  # take last timestep
        out = self.fc(out)
        return out

# Training function
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Val Accuracy: {accuracy:.2f}%")

    return model

if __name__ == "__main__":
    # Example usage - dummy data for demonstration
    seq_len = 30
    feature_dim = 10
    num_classes = 12
    num_samples = 1000

    # Random dataset for demo
    X = np.random.randn(num_samples, seq_len, feature_dim)
    y = np.random.randint(0, num_classes, size=num_samples)

    # Split into train/val
    split = int(0.8 * num_samples)
    train_ds = GestureDataset(X[:split], y[:split])
    val_ds = GestureDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = GestureLSTM(input_dim=feature_dim, output_dim=num_classes)
    trained_model = train_model(model, train_loader, val_loader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu')

    torch.save(trained_model.state_dict(), "gesture_lstm_model.pth")
