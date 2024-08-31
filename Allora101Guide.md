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
#### Optional: RPC :  https://sentries-rpc.testnet-1.testnet.allora.network/
```bash
{
    "wallet": {
        "addressKeyName": "test",
        "addressRestoreMnemonic": "<your mnemoric phase>",
        "alloraHomeDir": "/root/.allorad",
        "gas": "1000000",
        "gasAdjustment": 1.0,
        "nodeRpc": "https://allora-rpc.testnet-1.testnet.allora.network/",
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
from flask import Flask, Response
import requests
import json
import pandas as pd
import torch
import random

# create our Flask app
app = Flask(__name__)
        
def get_simple_price(token):
    base_url = "https://api.coingecko.com/api/v3/simple/price?ids="
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}&vs_currencies=usd"
        return url
    else:
        raise ValueError("Unsupported token") 
               
# define our endpoint
@app.route("/inference/<string:token>")
def get_inference(token):

    try:
      
        url = get_simple_price(token)
        headers = {
          "accept": "application/json",
          "x-cg-demo-api-key": "CG-XXXXXXXXXXXXXXXXXXXXXXXX" # replace with your API key
        }
    
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
          data = response.json()
          if token == 'BTC':
            price1 = data["bitcoin"]["usd"]*1.02
            price2 = data["bitcoin"]["usd"]*0.99
          if token == 'ETH':
            price1 = data["ethereum"]["usd"]*1.02
            price2 = data["ethereum"]["usd"]*0.98
          if token == 'SOL':
            price1 = data["solana"]["usd"]*1.02
            price2 = data["solana"]["usd"]*0.98
          if token == 'BNB':
            price1 = data["binancecoin"]["usd"]*1.02
            price2 = data["binancecoin"]["usd"]*0.98
          if token == 'ARB':
            price1 = data["arbitrum"]["usd"]*1.02
            price2 = data["arbitrum"]["usd"]*0.98
          random_float = str(round(random.uniform(price1, price2), 2))
        return random_float
    except Exception as e:
       # return Response(json.dumps({"pipeline error": str(e)}), status=500, mimetype='application/json')
        url = get_simple_price(token)
        headers = {
          "accept": "application/json",
          "x-cg-demo-api-key": "CG-XXXXXXXXXXXXXXXXXXXXXX # replace with your API key
        }
    
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
          data = response.json()
          if token == 'BTC':
            price1 = data["bitcoin"]["usd"]*1.02
            price2 = data["bitcoin"]["usd"]*0.98
          if token == 'ETH':
            price1 = data["ethereum"]["usd"]*1.02
            price2 = data["ethereum"]["usd"]*0.98
          if token == 'SOL':
            price1 = data["solana"]["usd"]*1.02
            price2 = data["solana"]["usd"]*0.98
          if token == 'BNB':
            price1 = data["binancecoin"]["usd"]*1.02
            price2 = data["binancecoin"]["usd"]*0.98
          if token == 'ARB':
            price1 = data["arbitrum"]["usd"]*1.02
            price2 = data["arbitrum"]["usd"]*0.98
          random_float = str(round(random.uniform(price1, price2), 2))
        return random_float

    
# run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

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
```
## Step 6: Build
```bash
docker compose up --build -d
```

HOW TO UPGRADE?
```bash
rm -rf upgrade-model.sh
wget https://raw.githubusercontent.com/0xScraipa/0gx/main/upgrade-model.sh && chmod +x upgrade-model.sh && ./upgrade-model.sh
```


## Check your wallet here: http://worker-tx.nodium.xyz/








