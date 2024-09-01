from flask import Flask, Response
import xgboost as xgb
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# Load the XGBoost model
model = xgb.Booster()
model.load_model('xgboost_model_tuned.json')

# Load the saved scaler
scaler = joblib.load('scaler.save')

def get_binance_url(symbol="ETHUSDT", interval="1m", limit=1000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

@app.route("/inference/<string:token>")
def get_inference(token):
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
        df["symbol"] = symbol
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Feature engineering
        df["price_change"] = df["close"] - df["open"]
        df["volatility"] = df["high"] - df["low"]
        df["symbol_encoded"] = pd.factorize(df["symbol"])[0]
        df["moving_average_10"] = df["close"].rolling(window=10).mean().fillna(df["close"].mean())
        df["moving_average_30"] = df["close"].rolling(window=30).mean().fillna(df["close"].mean())

        # Prepare the data for the XGBoost model
        features = df[["price_change", "volatility", "volume", "symbol_encoded", "moving_average_10", "moving_average_30"]]

        # Scale the features using the loaded scaler
        X_scaled = scaler.transform(features)

        # Convert to DMatrix
        dmatrix = xgb.DMatrix(X_scaled)

        # Make predictions
        try:
            forecast = model.predict(dmatrix)
            forecast_mean = forecast.mean()  # Calculate the mean prediction
            return Response(str(forecast_mean), status=200, mimetype='text/plain')
        except Exception as e:
            return Response(str(e), status=500, mimetype='text/plain')
    else:
        return Response(json.dumps({"Failed to retrieve data from Binance API": str(response.text)}), 
                        status=response.status_code, 
                        mimetype='application/json')

# Run Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
