import MetaTrader5 as mt5
import time

# تنظیمات
SYMBOL = "XAUUSD-VIP"
CHECK_INTERVAL = 2  # ثانیه
PRICE_UNIT = 1.5    # یک دلار
DEBUG = True

# اتصال به متاتریدر
if not mt5.initialize():
    print("❌ خطا در اتصال به متاتریدر:", mt5.last_error())
    quit()
print("✅ اتصال برقرار شد")

while True:
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) == 0:
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
        price = mt5.symbol_info_tick(SYMBOL).bid if is_buy else mt5.symbol_info_tick(SYMBOL).ask

        # محاسبه 1/4 مسیر تا TP
        tp_distance = abs(tp - entry)
        quarter_tp = entry + 0.05 * tp_distance if is_buy else entry - 0.05 * tp_distance

        if DEBUG:
            print(f"📈 قیمت: {price:.2f} | ورود: {entry:.2f} | TP: {tp:.2f} | SL فعلی: {sl:.2f} | ¼ TP: {quarter_tp:.2f}")

        # اگر قیمت هنوز به 1/4 نرسیده
        if (is_buy and price <= quarter_tp) or (not is_buy and price >= quarter_tp):
            if DEBUG:
                print("⏳ هنوز به ¼ مسیر نرسیده‌ایم.")
            continue

        # فاصله از 1/4 مسیر تا قیمت جاری
        move_from_qtp = abs(price - quarter_tp)
        units_moved = int(move_from_qtp // PRICE_UNIT)

        new_sl = quarter_tp + units_moved * PRICE_UNIT if is_buy else quarter_tp - units_moved * PRICE_UNIT
        if units_moved == 0:
            new_sl = quarter_tp

        # بررسی اینکه آیا SL باید تغییر کند؟
        if (is_buy and new_sl > sl) or (not is_buy and new_sl < sl):
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "tp": tp,
                "symbol": SYMBOL
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ SL تغییر کرد: {new_sl:.2f}")
            else:
                print(f"❌ تغییر SL شکست خورد: {result.retcode}")
        else:
            if DEBUG:
                print("⚠️ SL نیاز به تغییر ندارد.")

    time.sleep(CHECK_INTERVAL)
