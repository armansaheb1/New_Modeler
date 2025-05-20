import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Configuration
SEED = 42
SEQ_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 100
TARGET_COL = 'CLOSE'
torch.manual_seed(SEED)
np.random.seed(SEED)

class PriceDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length, 3]  # Close price at position 3
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load and preprocess data
df = pd.read_csv('your_data.txt', sep='\t', parse_dates={'datetime': ['DATE', 'TIME']})
df = df.set_index('datetime').sort_index()

# Add technical indicators
df.ta.log_return(cumulative=True, append=True)
df.ta.ema(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.macd(append=True)
df.ta.bbands(length=20, append=True)
df.ta.stoch(append=True)
df.ta.adx(length=14, append=True)
df.ta.vwap(append=True)
df.ta.cci(length=20, append=True)
df.ta.obv(append=True)
df = df.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)

# Prepare features and target
features = [col for col in df.columns if col != TARGET_COL]
X = df[features + [TARGET_COL]].values
y = df[TARGET_COL].values.reshape(-1, 1)

# Train-test split
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
full_scaled = scaler.transform(X)

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

sequences = create_sequences(full_scaled, SEQ_LENGTH)
targets = full_scaled[SEQ_LENGTH:, 3]

# Split sequences
test_start_idx = split_idx - SEQ_LENGTH
train_seq = sequences[:test_start_idx]
test_seq = sequences[test_start_idx:]
train_targets = targets[:test_start_idx]
test_targets = targets[test_start_idx:]

# Create datasets
train_dataset = PriceDataset(train_seq, SEQ_LENGTH)
test_dataset = PriceDataset(test_seq, SEQ_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = LSTMModel(
    input_size=X_train.shape[1],
    hidden_size=128,
    num_layers=2
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.6f} | Test Loss: {test_loss/len(test_loader):.6f}')

# Final predictions
with torch.no_grad():
    test_inputs = torch.FloatTensor(test_seq)
    predictions = model(test_inputs).numpy().flatten()

# Inverse scaling
dummy = np.zeros((len(predictions), full_scaled.shape[1]))
dummy[:, 3] = predictions
pred_prices = scaler.inverse_transform(dummy)[:, 3]

dummy[:, 3] = test_targets
true_prices = scaler.inverse_transform(dummy)[:, 3]

results = pd.DataFrame({
    'Actual': true_prices,
    'Predicted': pred_prices
})
print(results.tail())

# Save model
torch.save(model.state_dict(), 'price_predictor.pth')