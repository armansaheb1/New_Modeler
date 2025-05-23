import MetaTrader5 as mt5
import time

# تنظیمات
SYMBOL = "XAUUSD-VIP"
CHECK_INTERVAL = 2  # ثانیه
DEBUG = True

# اتصال به متاتریدر
if not mt5.initialize():
    print("❌ خطا در اتصال به متاتریدر:", mt5.last_error())
    quit()
print("✅ اتصال برقرار شد")

# دریافت دقت اعشار نماد
symbol_info = mt5.symbol_info(SYMBOL)
digits = symbol_info.digits if symbol_info else 2

while True:
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        if DEBUG:
            print("⏳ معامله فعالی نیست.")
        time.sleep(CHECK_INTERVAL)
        continue

    for pos in positions:
        ticket = pos.ticket
        entry = pos.price_open
        tp = pos.tp
        sl = pos.sl
        type_ = pos.type  # 0 = buy, 1 = sell
        is_buy = (type_ == mt5.ORDER_TYPE_BUY)

        tick = mt5.symbol_info_tick(SYMBOL)
        price = tick.bid if is_buy else tick.ask

        # محاسبه مسیر و ¼ آن
        tp_distance = abs(tp - entry)
        quarter_distance = tp_distance * 0.15
        quarter_distance2 = tp_distance * 0.08
        quarter_tp = entry + quarter_distance if is_buy else entry - quarter_distance

        if DEBUG:
            print(f"📈 قیمت: {price:.2f} | ورود: {entry:.2f} | TP: {tp:.2f} | SL فعلی: {sl:.2f} | ¼ TP: {quarter_tp:.2f}")

        # اگر قیمت هنوز به ¼ مسیر نرسیده
        if (is_buy and price <= entry + quarter_distance2 if is_buy else entry - quarter_distance2) or (not is_buy and price >= entry + quarter_distance2 if is_buy else entry - quarter_distance2):
            if DEBUG:
                print("⏳ هنوز به ¼ مسیر نرسیده‌ایم.")
            continue

        # SL موردنظر: همیشه quarter_distance عقب‌تر از قیمت فعلی
        new_sl = price - quarter_distance if is_buy else price + quarter_distance
        new_sl = round(new_sl, 2)

        # فقط اگر SL بهبود یافته است، ارسال کن
        if (is_buy and new_sl > sl) or (not is_buy and new_sl < sl):
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "tp": tp,
                "symbol": SYMBOL,
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ SL بروزرسانی شد: {new_sl:.2f}")
            else:
                print(f"❌ تغییر SL شکست خورد: {result.retcode}")
        else:
            if DEBUG:
                print("⚠️ نیازی به بروزرسانی SL نیست.")

    time.sleep(CHECK_INTERVAL)
