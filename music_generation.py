import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Dataset of chord sequences
class ChordDataset(Dataset):
    def __init__(self, sequences, chord_to_idx):
        self.sequences = sequences  # list of chord sequences (list of chord strings)
        self.chord_to_idx = chord_to_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = [self.chord_to_idx[ch] for ch in seq[:-1]]
        target_seq = [self.chord_to_idx[ch] for ch in seq[1:]]
        return torch.LongTensor(input_seq), torch.LongTensor(target_seq)

# RNN model for chord prediction
class ChordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super(ChordRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.fc(out)
        return out, hidden

def train_chord_rnn(model, dataloader, epochs=30, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    return model

def generate_chords(model, start_chord, chord_to_idx, idx_to_chord, length=20, device='cpu'):
    model.eval()
    model.to(device)
    input_idx = torch.LongTensor([[chord_to_idx[start_chord]]]).to(device)
    hidden = None
    generated = [start_chord]

    for _ in range(length):
        output, hidden = model(input_idx, hidden)
        probs = torch.softmax(output[0, -1], dim=0).detach().cpu().numpy()
        next_idx = np.random.choice(len(probs), p=probs)
        next_chord = idx_to_chord[next_idx]
        generated.append(next_chord)
        input_idx = torch.LongTensor([[next_idx]]).to(device)

    return generated

if __name__ == "__main__":
    # Example chords and sequences for training
    chords = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    chord_to_idx = {ch: i for i, ch in enumerate(chords)}
    idx_to_chord = {i: ch for ch, i in chord_to_idx.items()}

    # Example chord sequences (should be replaced by real data)
    sequences = [
        ["C", "F", "G", "C"],
        ["Am", "Dm", "G", "C"],
        ["C", "Em", "F", "G"],
        ["F", "G", "Em", "Am"],
        ["G", "F", "C", "Dm"],
    ] * 100  # replicate for more data

    dataset = ChordDataset(sequences, chord_to_idx)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ChordRNN(vocab_size=len(chords))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_chord_rnn(model, dataloader, epochs=20, device=device)

    # Generate chord progression
    start = "C"
    generated_seq = generate_chords(model, start, chord_to_idx, idx_to_chord, length=30, device=device)
    print("Generated chord progression:")
    print(" → ".join(generated_seq))

    # Save model weights
    torch.save(model.state_dict(), "chord_rnn_model.pth")
