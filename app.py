import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from flask import Flask, Response, json

app = Flask(__name__)

# Define the BiRNN model with the correct architecture
class BiRNNModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(BiRNNModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)  # *2 because of bidirectional

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size)  # *2 for bidirection
        rnn_out, _ = self.rnn(input_seq, h_0)
        predictions = self.linear(rnn_out[:, -1])
        return predictions

# Initialize the model with the same architecture as during training
model = BiRNNModel(input_size=1, hidden_layer_size=115, output_size=1, num_layers=2, dropout=0.3)
model.load_state_dict(torch.load("birnn_model_optimized.pth", weights_only=True))
model.eval()

# Function to fetch historical data from Binance
def get_binance_url(symbol="ETHUSDT", interval="1m", limit=1000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

@app.route("/inference/<string:token>")
def get_inference(token):
    if model is None:
        return Response(json.dumps({"error": "Model is not available"}), status=500, mimetype='application/json')

    symbol_map = {
        'ETH': 'ETHUSDT',
        'BTC': 'BTCUSDT',
        'BNB': 'BNBUSDT',
        'SOL': 'SOLUSDT',
        'ARB': 'ARBUSDT'
    }

    token = token.upper()
    if token in symbol_map:
        symbol = symbol_map[token]
    else:
        return Response(json.dumps({"error": "Unsupported token"}), status=400, mimetype='application/json')

    url = get_binance_url(symbol=symbol)
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

        # Adjust the number of rows based on the symbol
        if symbol in ['BTCUSDT', 'SOLUSDT']:
            df = df.tail(10)  # Use last 10 minutes of data
        else:
            df = df.tail(20)  # Use last 20 minutes of data

        # Prepare data for the BiRNN model
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))

        seq = torch.FloatTensor(scaled_data).view(1, -1, 1)

        # Make prediction
        with torch.no_grad():
            y_pred = model(seq)

        # Inverse transform the prediction to get the actual price
        predicted_price = scaler.inverse_transform(y_pred.numpy())

        # Round the predicted price to 2 decimal places
        rounded_price = round(predicted_price.item(), 2)

        # Return the rounded price as a string
        return Response(str(rounded_price), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": response.text}), 
                        status=response.status_code, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
