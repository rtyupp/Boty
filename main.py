import os
import io
import time
import logging
import requests
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timezone
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

# ================= إعدادات البوت =================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================= Headers =================
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

# ================= مصادر البيانات =================
def fetch_from_yahoo(symbol="GC=F", interval="5m"):
    """المصدر الأول: Yahoo Finance"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {'interval': interval, 'range': '5d'}
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'chart' not in data or not data['chart']['result']:
            return None, None
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s', utc=True),
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume']
        })
        df.set_index('Date', inplace=True)
        df.dropna(inplace=True)
        
        last_update = datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)
        return df, last_update
        
    except Exception as e:
        logging.error(f"Yahoo Finance error: {e}")
        return None, None

def fetch_from_twelve_data(api_key=None):
    """المصدر الثاني: Twelve Data API (مجاني - يحتاج تسجيل)"""
    if not api_key:
        return None, None
    
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': 'XAU/USD',
            'interval': '5min',
            'outputsize': 100,
            'apikey': api_key
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'values' not in data:
            return None, None
        
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        df = df[::-1]  # عكس الترتيب (الأحدث أولاً)
        last_update = df.index[-1]
        
        return df, last_update
        
    except Exception as e:
        logging.error(f"Twelve Data error: {e}")
        return None, None

def fetch_from_exchangerate():
    """المصدر الثالث: ExchangeRate API (سعر فوري فقط)"""
    try:
        url = "https://api.exchangerate-api.com/v4/latest/XAU"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'rates' in data and 'USD' in data['rates']:
            price = data['rates']['USD']
            logging.info(f"✅ سعر الذهب الفوري: ${price:.2f}")
            return price, datetime.now(timezone.utc)
        
        return None, None
        
    except Exception as e:
        logging.error(f"ExchangeRate API error: {e}")
        return None, None

def get_gold_data():
    """جلب البيانات من أفضل مصدر متاح"""
    logging.info("🔄 محاولة جلب البيانات من Yahoo Finance...")
    df, last_update = fetch_from_yahoo("GC=F", "5m")
    
    if df is not None and len(df) >= 30:
        # التحقق من حداثة البيانات (آخر ساعة)
        now = datetime.now(timezone.utc)
        if (now - last_update).total_seconds() < 3600:
            logging.info(f"✅ بيانات Yahoo Finance حديثة ({len(df)} شمعة)")
            return df, last_update, "Yahoo Finance (Futures)"
        else:
            logging.warning("⚠️ بيانات Yahoo قديمة جداً")
    
    # محاولة Twelve Data إذا كان API Key متوفر
    api_key = os.environ.get("TWELVE_DATA_API_KEY")
    if api_key:
        logging.info("🔄 محاولة Twelve Data...")
        df, last_update = fetch_from_twelve_data(api_key)
        if df is not None and len(df) >= 30:
            logging.info(f"✅ بيانات Twelve Data ({len(df)} شمعة)")
            return df, last_update, "Twelve Data (Spot)"
    
    # إذا فشل كل شيء، نرجع بيانات Yahoo حتى لو قديمة
    if df is not None:
        logging.warning("⚠️ استخدام بيانات قديمة من Yahoo")
        return df, last_update, "Yahoo Finance (قديمة)"
    
    return None, None, None

# ================= التحليل الفني =================
def analyze_trend(data):
    """تحليل الاتجاه"""
    if data is None or len(data) < 30:
        return "⚠️ بيانات غير كافية", 0, 0, 0, 0

    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['Close'], 14)
    
    last_price = float(data['Close'].iloc[-1])
    ema_9 = float(data['EMA_9'].iloc[-1])
    ema_21 = float(data['EMA_21'].iloc[-1])
    rsi = float(data['RSI'].iloc[-1])
    
    trend_score = 0
    
    if ema_9 > ema_21:
        trend_score += 1
    else:
        trend_score -= 1
    
    if last_price > ema_9:
        trend_score += 1
    else:
        trend_score -= 1
    
    if rsi > 70:
        trend_score -= 1
    elif rsi < 30:
        trend_score += 1
    
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
    """حساب RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ================= إنشاء الشارت =================
def generate_chart(data, prediction):
    """إنشاء الشارت"""
    if data is None:
        return None
    
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
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
        "✅ بيانات من مصادر متعددة\n"
        "✅ عرض وقت آخر تحديث\n"
        "✅ تحليل فني مع RSI\n\n"
        "اضغط الزر أدناه للبدء 👇",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("⏳ جاري التحليل...")
    
    await query.edit_message_text("🔄 *جاري جلب بيانات الذهب من مصادر متعددة...*", parse_mode='Markdown')
    
    # جلب البيانات
    data, last_update, source = get_gold_data()
    
    if data is None:
        await query.edit_message_text(
            "❌ *فشل في جلب البيانات من جميع المصادر.*\n\n"
            "الأسباب المحتملة:\n"
            "• السوق مغلق (عطلة نهاية الأسبوع)\n"
            "• حظر مؤقت من APIs\n\n"
            "حاول مرة أخرى لاحقاً.",
            parse_mode='Markdown'
        )
        return
    
    # التحليل
    prediction, confidence, ema_9, ema_21, rsi = analyze_trend(data)
    last_price = float(data['Close'].iloc[-1])
    
    # تنسيق وقت التحديث
    update_time_str = last_update.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # التحقق من حداثة البيانات
    now = datetime.now(timezone.utc)
    age_minutes = int((now - last_update).total_seconds() / 60)
    
    if age_minutes > 60:
        freshness_warning = f"\n⚠️ *تحذير:* البيانات عمرها {age_minutes} دقيقة (قديمة)"
    else:
        freshness_warning = f"\n✅ البيانات محدثة (عمرها {age_minutes} دقيقة)"
    
    # إنشاء الشارت
    chart_img = generate_chart(data, prediction)
    
    # بناء الرسالة
    message_text = (
        f"💰 *تحليل الذهب (XAU/USD)*\n"
        f"⏱ *الفريم:* 5 دقائق\n\n"
        f"📈 *السعر:* `${last_price:.2f}$\n"
        f"🎯 *التوقع:* *{prediction}*\n"
        f"💪 *قوة الإشارة:* {confidence}%\n\n"
        f"📊 *المؤشرات:*\n"
        f"• EMA 9: `{ema_9:.2f}`\n"
        f"• EMA 21: `{ema_21:.2f}`\n"
        f"• RSI (14): `{rsi:.1f}`\n\n"
        f"🕐 *آخر تحديث:* `{update_time_str}`\n"
        f"📡 *المصدر:* {source}"
        f"{freshness_warning}\n\n"
        f"⚡ _مناسب للـ Scalping_"
    )
    
    keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data='refresh')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # إرسال الصورة
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
            f"{message_text}\n\n⚠️ *تعذر إنشاء الشارت.*",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )

# ================= تشغيل البوت =================
def main():
    if not BOT_TOKEN:
        logging.error("❌ BOT_TOKEN غير موجود!")
        return
    
    logging.info("🚀 بدء تشغيل البوت...")
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_button))
    
    logging.info("✅ البوت يعمل الآن...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
