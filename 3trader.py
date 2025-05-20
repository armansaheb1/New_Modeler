import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import ta
import time

# اتصال به متاتریدر
if not mt5.initialize():
    print("❌ MetaTrader5 initialization failed")
    quit()

symbol = "XAUUSD-VIP"
timeframe = mt5.TIMEFRAME_M30
threshold = 1.5  # اختلاف پیش‌بینی با قیمت فعلی

# لود مدل PyTorch
class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super(PricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model_input_size = 12  # باید با تعداد ویژگی‌ها سازگار باشد
model = PricePredictor(model_input_size)
model.load_state_dict(torch.load("model.pt"))
model.eval()

scaler = MinMaxScaler()

def has_open_position():
    positions = mt5.positions_get(symbol=symbol)
    return positions is not None and len(positions) > 0

def get_last_candles(n=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) < n:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_features(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_bbw'] = bb.bollinger_wband()
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['atr'] = atr
    df = df.dropna()
    return df

def prepare_input(df):
    features = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'rsi', 'ema', 'macd', 'bb_bbw', 'atr']
    df = df[features]
    df = df.dropna()
    data = scaler.fit_transform(df)
    X = torch.tensor(data[-3:], dtype=torch.float32).unsqueeze(0)
    return X, df['close'].iloc[-1], df['atr'].iloc[-1]

def get_dynamic_lot():
    account_info = mt5.account_info()
    if account_info is None:
        return 0.01
    balance = account_info.balance
    lot = round(balance / 5000, 2)
    return max(0.01, lot)

def open_trade(predicted_price, current_price, direction, atr):
    price = mt5.symbol_info_tick(symbol).ask if direction == 'buy' else mt5.symbol_info_tick(symbol).bid
    sl = price - atr if direction == 'buy' else price + atr
    tp = price + atr * 1.5 if direction == 'buy' else price - atr * 1.5
    order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
    lot = get_dynamic_lot()

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": round(sl, 3),
        "tp": round(tp, 3),
        "deviation": 10,
        "magic": 123456,
        "comment": "AutoTrade by LSTM",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order send failed: {result.retcode}")
    else:
        print(f"✅ Trade opened: {direction.upper()}, Lot: {lot}, SL: {sl:.3f}, TP: {tp:.3f}")

# حلقه اجرای یک‌باره (یا می‌توانید در حلقه قرار دهید)
df = get_last_candles()
if df is not None:
    df = calculate_features(df)
    if len(df) >= 3:
        X, current_price, atr = prepare_input(df)
        with torch.no_grad():
            predicted = model(X).item()

        print(f"📊 Current Price: {current_price:.2f} | Predicted: {predicted:.2f} | ATR: {atr:.4f}")

        if not has_open_position():
            if predicted > current_price + threshold:
                open_trade(predicted, current_price, direction='buy', atr=atr)
            elif predicted < current_price - threshold:
                open_trade(predicted, current_price, direction='sell', atr=atr)
            else:
                print("ℹ️ No valid signal (threshold not met)")
        else:
            print("ℹ️ Position already open")
else:
    print("❌ Failed to retrieve candles")

mt5.shutdown()
