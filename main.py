import os
import io
import time
import logging
import requests
import pandas as pd
import mplfinance as mpf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

# ================= إعدادات البوت =================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# رموز الذهب البديلة (نجربها بالترتيب)
GOLD_SYMBOLS = ["GC=F", "XAUUSD=X", "GLD"]

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================= Headers تحاكي المتصفح (مهم جداً!) =================
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
}

# ================= دوال جلب البيانات =================
def fetch_gold_data_yahoo(symbol, interval='5m', period='5d'):
    """جلب البيانات مباشرة من Yahoo Finance API"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        'interval': interval,
        'range': period,
        'includePrePost': 'false',
    }
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            logging.warning(f"لا توجد بيانات للرمز {symbol}")
            return None
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        indicators = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Open': indicators['open'],
            'High': indicators['high'],
            'Low': indicators['low'],
            'Close': indicators['close'],
            'Volume': indicators['volume']
        })
        
        df.set_index('Date', inplace=True)
        df.dropna(inplace=True)
        
        logging.info(f"✅ تم جلب {len(df)} شمعة للرمز {symbol}")
        return df
        
    except Exception as e:
        logging.error(f"خطأ في جلب البيانات من {symbol}: {e}")
        return None

def get_gold_data():
    """جلب بيانات الذهب مع retry ورموز بديلة"""
    for symbol in GOLD_SYMBOLS:
        logging.info(f"🔄 نجرب الرمز: {symbol}")
        data = fetch_gold_data_yahoo(symbol, interval='5m', period='5d')
        if data is not None and len(data) >= 30:
            return data
        time.sleep(1)  # انتظار ثانية قبل المحاولة التالية
    
    logging.error("❌ فشلت جميع المحاولات في جلب البيانات")
    return None

# ================= التحليل الفني =================
def analyze_trend(data):
    """تحليل الاتجاه لفريم 5 دقائق"""
    if data is None or len(data) < 30:
        return "⚠️ بيانات غير كافية", 0, 0, 0

    # حساب المؤشرات
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)
    
    last_price = float(data['Close'].iloc[-1])
    ema_9 = float(data['EMA_9'].iloc[-1])
    ema_21 = float(data['EMA_21'].iloc[-1])
    rsi = float(data['RSI'].iloc[-1])
    
    # تحليل متقدم
    trend_score = 0
    
    # 1. اتجاه EMA
    if ema_9 > ema_21:
        trend_score += 1
    else:
        trend_score -= 1
    
    # 2. موقع السعر
    if last_price > ema_9:
        trend_score += 1
    else:
        trend_score -= 1
    
    # 3. RSI
    if rsi > 70:
        trend_score -= 1  # تشبع شرائي
    elif rsi < 30:
        trend_score += 1  # تشبع بيعي
    
    # تحديد التوقع
    if trend_score >= 2:
        prediction = "صعود قوي 🟢"
        confidence = 85
    elif trend_score == 1:
        prediction = "صعود ضعيف 🟢"
        confidence = 65
    elif trend_score <= -2:
        prediction = "هبوط قوي 🔴"
        confidence = 85
    elif trend_score == -1:
        prediction = "هبوط ضعيف 🔴"
        confidence = 65
    else:
        prediction = "تذبذب / عرضي "
        confidence = 50
    
    return prediction, confidence, ema_9, ema_21, rsi

def calculate_rsi(series, period=14):
    """حساب مؤشر RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ================= إنشاء الشارت =================
def generate_chart(data, prediction):
    """إنشاء الشارت في الذاكرة"""
    if data is None:
        return None
    
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
    # ألوان حسب التوقع
    if 'صعود' in prediction:
        ema9_color = '#00FF00'
    elif 'هبوط' in prediction:
        ema9_color = '#FF0000'
    else:
        ema9_color = '#FFA500'
    
    add_plots = [
        mpf.make_addplot(data['EMA_9'], color=ema9_color, width=1.5),
        mpf.make_addplot(data['EMA_21'], color='#FFA500', width=1.5)
    ]
    
    img_buffer = io.BytesIO()
    
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#eeeeee')
    
    mpf.plot(
        data, type='candle', addplot=add_plots,
        title=f'Gold (XAU/USD) - 5min | {prediction}',
        style=s, figsize=(12, 7), savefig=img_buffer,
        tight_layout=True
    )
    
    img_buffer.seek(0)
    return img_buffer

# ================= دوال تيليجرام =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 شارت + تحليل F5", callback_data='chart_5m')],
        [InlineKeyboardButton("🔄 تحديث فوري", callback_data='refresh')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 *بوت الذهب الاحترافي - فريم 5 دقائق*\n\n"
        "✅ تحليل فني لحظي\n"
        "✅ توقعات صعود/هبوط مع RSI\n"
        "✅ شارت مباشر مع EMA 9 & 21\n\n"
        "اضغط الزر أدناه للبدء 👇",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("⏳ جاري التحليل...")
    
    await query.edit_message_text("🔄 *جاري جلب بيانات الذهب وتحليلها...*", parse_mode='Markdown')
    
    # جلب البيانات
    data = get_gold_data()
    if data is None:
        await query.edit_message_text(
            "❌ *فشل في جلب البيانات من Yahoo Finance.*\n\n"
            "الأسباب المحتملة:\n"
            "• حظر مؤقت من Yahoo\n"
            "• مشاكل في الاتصال\n\n"
            "حاول مرة أخرى بعد 5 دقائق.",
            parse_mode='Markdown'
        )
        return
    
    # التحليل
    prediction, confidence, ema_9, ema_21, rsi = analyze_trend(data)
    last_price = float(data['Close'].iloc[-1])
    
    # إنشاء الشارت
    chart_img = generate_chart(data, prediction)
    
    # بناء الرسالة
    message_text = (
        f"💰 *تحليل الذهب (XAU/USD)*\n"
        f"⏱ *الفريم:* 5 دقائق\n\n"
        f"📈 *السعر الحالي:* `${last_price:.2f}$`\n"
        f"🎯 *التوقع:* *{prediction}*\n"
        f" *قوة الإشارة:* {confidence}%\n\n"
        f"📊 *المؤشرات:*\n"
        f"• EMA 9: `{ema_9:.2f}`\n"
        f"• EMA 21: `{ema_21:.2f}`\n"
        f"• RSI (14): `{rsi:.1f}`\n\n"
        f"⚡ _الفريم 5 دقائق سريع - مناسب للـ Scalping_"
    )
    
    keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data='refresh')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # إرسال الصورة مع النص
    if chart_img:
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=chart_img,
            caption=message_text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        try:
            await query.message.delete()
        except:
            pass
    else:
        await query.edit_message_text(
            f"{message_text}\n\n⚠️ *تعذر إنشاء صورة الشارت.*",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

# ================= تشغيل البوت =================
def main():
    if not BOT_TOKEN:
        logging.error(" BOT_TOKEN غير موجود في Environment Variables!")
        return
    
    logging.info("🚀 بدء تشغيل البوت...")
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_button))
    
    logging.info("✅ البوت يعمل الآن ويراقب الذهب...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
