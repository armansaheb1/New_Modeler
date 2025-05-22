import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from datetime import datetime, timedelta

# تنظیمات کلی
SYMBOL = "XAUUSD-VIP"
TIMEFRAME = mt5.TIMEFRAME_M30
N_REQUIRED = 15  # حداقل تعداد کندل مورد نیاز
TP_COEF = 0.9   # ضریب محافظه‌کارانه برای TP
SL_BUFFER = 1.1 # فاصله احتیاطی برای SL
VOLUME_RATIO = 0.0002  # حجم معامله نسبت به موجودی (یک پنج هزارم)

# مدل LSTM (مثل مدل شما)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def fetch_candles(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 100)
    if rates is None or len(rates) < n:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'tick_volume']]

def enrich_features(df):
    df_feat = add_all_ta_features(
        df.copy(),
        open="open", high="high", low="low",
        close="close", volume="tick_volume",
        fillna=True
    )
    return df_feat

def load_model(input_size):
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load("price_predictor.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def prepare_features(df_feat, window=3):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat)
    features = []
    for i in range(window, len(df_feat)):
        features.append(scaled[i - window:i])
    features = torch.tensor(np.array(features), dtype=torch.float32)
    return features, scaler

def calculate_tp_sl(pred_prices, current_price):
    diffs = pred_prices - current_price
    positive_diffs = diffs[diffs > 0]
    negative_diffs = -diffs[diffs < 0]  # برای sell، قدرمطلق منفی‌ها

    if np.mean(diffs) > 0:
        direction = "buy"
        avg_increase = np.mean(positive_diffs) if len(positive_diffs) > 0 else 0.001
        tp = current_price + min((avg_increase * TP_COEF), 9)
        sl = current_price - min((avg_increase * TP_COEF * 3/4), 2)
    else:
        direction = "sell"
        avg_decrease = np.mean(negative_diffs) if len(negative_diffs) > 0 else 0.001
        tp = current_price - min((avg_decrease * TP_COEF), 9)
        sl = current_price + min((avg_decrease * TP_COEF * 3/4), 2)

    return direction, tp, sl
def get_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    elif len(positions) == 0:
        return None
    else:
        return positions[0]  # فقط یک پوزیشن داریم

def place_order(direction, volume, price_tp, price_sl):
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None or not symbol_info.visible:
        print(f"❌ نماد {SYMBOL} در متاتریدر قابل مشاهده نیست.")
        return False


    deviation = 20  # انحراف مجاز قیمت (پیپ) برای ارسال سفارش

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(SYMBOL).ask if direction == "buy" else mt5.symbol_info_tick(SYMBOL).bid,
        "sl": price_sl,
        "tp": price_tp,
        "deviation": deviation,
        "magic": 123456,
        "comment": "LSTM Prediction Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ ارسال سفارش با خطا مواجه شد: {result.comment}")
        return False
    print(f"✅ سفارش {direction} با حجم {volume} با موفقیت ارسال شد. TP: {price_tp:.4f}, SL: {price_sl:.4f}")
    return True

if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ اتصال به متاتریدر5 برقرار نشد")
        mt5.shutdown()
        exit()

    df = fetch_candles(SYMBOL, TIMEFRAME, N_REQUIRED)
    if df is None:
        print(f"❌ تعداد کندل کافی دریافت نشد (کمتر از {N_REQUIRED} کندل)")
        mt5.shutdown()
        exit()

    df_feat = enrich_features(df)

    window = 3
    X, scaler = prepare_features(df_feat, window=window)
    input_size = X.shape[2]

    model = load_model(input_size)

    with torch.no_grad():
        pred = model(X)
        pred_prices = pred.numpy().flatten()

    scaler_close = MinMaxScaler()
    scaler_close.fit(df_feat[['close']])
    pred_inv = scaler_close.inverse_transform(pred_prices.reshape(-1,1)).flatten()

    current_price = df['close'].iloc[-1]
    direction, tp, sl = calculate_tp_sl(pred_inv, current_price)

    # محاسبه حجم معامله بر اساس موجودی
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ دریافت اطلاعات حساب ممکن نشد")
        mt5.shutdown()
        exit()
    volume = round(account_info.balance * VOLUME_RATIO, 2)
    volume = max(volume, 0.01)  # حداقل حجم 0.01 لات

    # بررسی وجود معامله باز
    pos = get_position(SYMBOL)
    def last_closed_loss_trade_direction(symbol):
        from time import gmtime, strftime, mktime

        start = datetime.fromtimestamp(mktime(gmtime())) - timedelta(days=2)
        end = datetime.fromtimestamp(mktime(gmtime())) + timedelta(days=30)

        deals = mt5.history_deals_get(start, end)

        if deals is None or len(deals) == 0:
            print("❌ هیچ معامله‌ای در تاریخچه پیدا نشد.")
            return None
        if deals[-1].profit < 0:
            return "sell" if deals[-1].type == 0 else "buy"
        return None
    
    def confirm_candle_strength(direction):
    # دریافت 100 کندل آخر
        candles = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
        
        # انتخاب دو کندل اخیر (بسته‌شده)
        last_closed = candles[-2]
        prev = candles[-3]
        
        # محاسبه بدنه کندل‌ها
        last_body = abs(last_closed['close'] - last_closed['open'])
        prev_body = abs(prev['close'] - prev['open'])

        # تعیین جهت کندل‌ها
        last_dir = 'buy' if last_closed['close'] > last_closed['open'] else 'sell'
        prev_dir = 'buy' if prev['close'] > prev['open'] else 'sell'

        direction = direction.lower()

        # بررسی شرایط قدرت کندل در جهت مورد نظر
        if direction == 'buy':
            if last_dir == prev_dir == 'buy':
                return True
            print('[Buy] Not strong enough')
            return last_dir == 'buy' and last_body > prev_body

        elif direction == 'sell':
            if last_dir == prev_dir == 'sell':
                return True
            print('[Sell] Not strong enough')
            return last_dir == 'sell' and last_body > prev_body

        # اگر ورودی اشتباه باشد
        print('[Error] Invalid direction input')
        return False
    last_loss_dir = last_closed_loss_trade_direction(SYMBOL)
    if pos is not None:
        print("❌ معامله باز موجود است، معامله جدید باز نمی‌شود.")
    
    elif last_loss_dir == direction:
        if confirm_candle_strength(direction):
            place_order(direction, volume, tp, sl)
        print(f"❌ آخرین معامله {direction} زیان‌ده بوده. منتظر معامله در جهت مخالف هستیم.")
    else:
        place_order(direction, volume, tp, sl)
    mt5.shutdown()
