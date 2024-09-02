import torch
import torch.nn as nn
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

# Define the enhanced model with LSTM layers
class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(EnhancedBiLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * 2)  # *2 for bidirectional and 2 timeframes

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)
        
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Function to fetch historical data from Binance
def get_binance_data(symbol="ETHUSDT", interval="1m", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)
        return df
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")

# Prepare the dataset
def prepare_dataset(symbols, sequence_length=10):
    all_data = []
    for symbol in symbols:
        df = get_binance_data(symbol)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
        for i in range(sequence_length, len(scaled_data) - 2):  # to account for the 20-minute prediction
            seq = scaled_data[i-sequence_length:i]
            label_10 = scaled_data[i+10] if i+10 < len(scaled_data) else scaled_data[-1]
            label_20 = scaled_data[i+20] if i+20 < len(scaled_data) else scaled_data[-1]
            label = torch.FloatTensor([label_10[0], label_20[0]])
            all_data.append((seq, label))
    return all_data, scaler

# Define the training process
def train_model(model, data, epochs=50, lr=0.001, sequence_length=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for seq, label in data:
            seq = torch.FloatTensor(seq).view(1, sequence_length, -1)
            label = label.view(1, -1)  # Ensure label has the shape [batch_size, 2]

            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data)}')

    torch.save(model.state_dict(), "enhanced_bilstm_model.pth")
    print("Model trained and saved as enhanced_bilstm_model.pth")

if __name__ == "__main__":
    # Define the model
    model = EnhancedBiLSTMModel(input_size=1, hidden_layer_size=115, output_size=1, num_layers=2, dropout=0.3)

    # Symbols to train on
    symbols = ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ARBUSDT']

    # Prepare data
    data, scaler = prepare_dataset(symbols)

    # Train the model
    train_model(model, data)
