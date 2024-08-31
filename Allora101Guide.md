# Allora Guide 

- You must need to buy a VPS for running Allora Worker
- You can buy from : Contabo
- You should buy VPS which is fulfilling all these requirements : 
```bash
Operating System : Ubuntu 22.04
CPU: Minimum of 1/2 core.
Memory: 2 to 4 GB.
Storage: SSD or NVMe with at least 5GB of space.
```
# Prerequisites
Before you start, ensure you have docker compose installed.
```bash
# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
docker version

# Install Docker-Compose
VER=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)

curl -L "https://github.com/docker/compose/releases/download/"$VER"/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

chmod +x /usr/local/bin/docker-compose
docker-compose --version

# Docker Permission to user
sudo groupadd docker
sudo usermod -aG docker $USER
```

Clean Old Docker
```
docker compose down -v
docker container prune
cd $HOME && rm -rf allora-huggingface-walkthrough
```

### Deployment - Read Carefully! 
## Step 1: 
```bash
git clone https://github.com/allora-network/allora-huggingface-walkthrough
cd allora-huggingface-walkthrough
```
## Step 2: 
```bash
cp config.example.json config.json
nano config.json
```

####  Edit addressKeyName & addressRestoreMnemonic / Copy & Paste Inside config.json
#### Optional: RPC :  https://allora-rpc.testnet-1.testnet.allora.network/
```bash
{
    "wallet": {
        "addressKeyName": "test",
        "addressRestoreMnemonic": "<your mnemoric phase>",
        "alloraHomeDir": "/root/.allorad",
        "gas": "1000000",
        "gasAdjustment": 1.0,
        "nodeRpc": "https://rpc.ankr.com/allora_testnet/",
        "maxRetries": 1,
        "delay": 1,
        "submitTx": false
    },
    "worker": [
        {
            "topicId": 1,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 4,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ETH"
            }
        },
        {
            "topicId": 3,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 6,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "BTC"
            }
        },
        {
            "topicId": 5,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 8,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "SOL"
            }
        },
        {
            "topicId": 7,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 2,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ETH"
            }
        },
        {
            "topicId": 8,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 3,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "BNB"
            }
        },
        {
            "topicId": 9,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 5,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ARB"
            }
        }
        
    ]
}
```
## Step 3: Export 
```bash
chmod +x init.config
./init.config
```
## Step 4: Edit App.py
- Register on Coingecko https://www.coingecko.com/en/developers/dashboard & Create Demo API KEY
- Copy & Replace API with your `UPSHOT API` -`COINGECKO API` , then save `Ctrl+X Y ENTER`.
```bash
nano app.py
```
```bash
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

```

## Step 5: Edit requirements.txt
```bash
nano requirements.txt
```
#- Copy & Paste, then save `Ctrl+X Y ENTER`.

```
flask[async]
gunicorn[gthread]
transformers[torch]
pandas
torch==2.0.1 
python-dotenv
requests==2.31.0
scikit-learn==1.3.1
numpy==1.25.2
joblib==1.3.2
```
## Step 6: Build
```bash
docker compose up --build -d
```
-------------------------------------------------------------

## Check your wallet here: http://worker-tx.nodium.xyz/








