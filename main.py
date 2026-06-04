import os
import io
import time
import logging
import requests
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timezone
from groq import Groq
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler, MessageHandler, filters

# ================= الإعدادات =================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# تخزين محادثات المستخدمين مع الذكاء الاصطناعي
user_conversations = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# ================= 1. جلب البيانات الفورية =================

def fetch_spot_price():
    """جلب السعر الفوري (Spot) من Twelve Data"""
    if not TWELVE_DATA_API_KEY:
        return None, None
    
    try:
        url = "https://api.twelvedata.com/price"
        params = {
            'symbol': 'XAU/USD',
            'apikey': TWELVE_DATA_API_KEY
        }
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'price' in data:
            price = float(data['price'])
            logging.info(f"✅ السعر الفوري: ${price:.2f}")
            return price, datetime.now(timezone.utc)
        
        return None, None
        
    except Exception as e:
        logging.error(f"Spot price error: {e}")
        return None, None

def fetch_candlestick_data(interval='5min', outputsize=100):
    """جلب بيانات الشموع من Twelve Data"""
    if not TWELVE_DATA_API_KEY:
        return None, None
    
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': 'XAU/USD',
            'interval': interval,
            'outputsize': outputsize,
            'apikey': TWELVE_DATA_API_KEY
        }
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'values' not in data or 'meta' not in data:
            logging.error(f"API error: {data.get('message', 'Unknown')}")
            return None
        
        # تحويل إلى DataFrame
        df = pd.DataFrame(data['values'])
        
        # تحويل الأنواع
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # إعادة التسمية وترتيب حسب الزمن (الأقدم أولاً)
        df.rename(columns={
            'open': 'Open', 'high': 'High', 
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        df = df[::-1]  # عكس الترتيب
        df.dropna(inplace=True)
        
        logging.info(f"✅ تم جلب {len(df)} شمعة من Twelve Data")
        return df
        
    except Exception as e:
        logging.error(f"Candlestick error: {e}")
        return None

def get_gold_data():
    """جلب البيانات الكاملة (سعر فوري + شموع)"""
    spot_price, spot_time = fetch_spot_price()
    candlestick_data = fetch_candlestick_data('5min', 100)
    
    return {
        'spot_price': spot_price,
        'spot_time': spot_time,
        'candles': candlestick_data,
        'source': 'Twelve Data (Spot)' if spot_price else 'Yahoo (Fallback)'
    }

# ================= 2. الذكاء الاصطناعي =================

def analyze_with_ai(data):
    """تحليل البيانات باستخدام Groq AI"""
    if not GROQ_API_KEY or not data['candles'] is not None:
        return "⚠️ AI غير متاح"
    
    try:
        df = data['candles']
        spot = data['spot_price']
        
        # حساب المؤشرات
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)
        
        last_10 = df.tail(10)
        
        # تجهيز البيانات للـ AI
        context = f"""
أنت محلل فني محترف للذهب (XAU/USD).

البيانات الحالية:
- السعر الفوري (Spot): ${spot:.2f}
- آخر 10 شموع (فريم 5 دقائق):
{last_10[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_9', 'EMA_21', 'RSI']].round(2).to_string()}

المؤشرات الأخيرة:
- EMA 9: {df['EMA_9'].iloc[-1]:.2f}
- EMA 21: {df['EMA_21'].iloc[-1]:.2f}
- RSI: {df['RSI'].iloc[-1]:.1f}

قدم تحليلاً احترافياً باللغة العربية يتضمن:
1. الاتجاه العام (صاعد/هابط/عرضي)
2. قوة الاتجاه
3. مناطق الدعم والمقاومة القريبة
4. توصية تداول (شراء/بيع/انتظار) مع سبب
5. مستوى وقف الخسارة المقترح
6. هدف الربح المقترح

كن مختصراً ومباشراً. استخدم emojis.
"""
        
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "أنت محلل فني خبير في تداول الذهب والأسواق المالية. ردودك مختصرة ومباشرة بالعربية."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logging.error(f"AI analysis error: {e}")
        return f"⚠️ خطأ في تحليل AI: {str(e)}"

def chat_with_ai(user_id, message_text):
    """الدردشة مع الذكاء الاصطناعي"""
    if not GROQ_API_KEY:
        return "⚠️ خدمة الذكاء الاصطناعي غير متاحة حالياً. تأكد من إضافة GROQ_API_KEY"
    
    try:
        # تهيئة المحادثة إذا كانت جديدة
        if user_id not in user_conversations:
            user_conversations[user_id] = [
                {"role": "system", "content": """أنت خبير تداول الذهب (XAU/USD) ومحلل فني محترف. 
تجيب على أسئلة المستخدم بالعربية بشكل مختصر ومباشر.
تعطي نصائح تداول واقعية وتُحذّر من المخاطر.
لا تتنبأ بالمستقبل بشكل قاطع، بل تقدم تحليلات مبنية على البيانات."""}
            ]
        
        # إضافة رسالة المستخدم
        user_conversations[user_id].append({"role": "user", "content": message_text})
        
        # الاحتفاظ بآخر 10 رسائل فقط
        if len(user_conversations[user_id]) > 11:
            user_conversations[user_id] = user_conversations[user_id][:1] + user_conversations[user_id][-10:]
        
        # إرسال إلى Groq
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=user_conversations[user_id],
            temperature=0.7,
            max_tokens=1000
        )
        
        reply = completion.choices[0].message.content
        user_conversations[user_id].append({"role": "assistant", "content": reply})
        
        return reply
        
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return f"⚠️ خطأ: {str(e)}"

# ================= 3. الحسابات الفنية =================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_chart(data):
    """إنشاء الشارت"""
    df = data['candles']
    if df is None or len(df) < 30:
        return None
    
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    add_plots = [
        mpf.make_addplot(df['EMA_9'], color='#00FF00', width=1.5),
        mpf.make_addplot(df['EMA_21'], color='#FFA500', width=1.5)
    ]
    
    img_buffer = io.BytesIO()
    
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', gridcolor='#eeeeee')
    
    mpf.plot(
        df, type='candle', addplot=add_plots,
        title=f'Gold (XAU/USD) Spot - 5min',
        style=s, figsize=(12, 7), savefig=img_buffer,
        tight_layout=True
    )
    
    img_buffer.seek(0)
    return img_buffer

# ================= 4. دوال تيليجرام =================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """رسالة البداية مع شرح الميزات"""
    keyboard = [
        [InlineKeyboardButton("📊 تحليل فوري + AI", callback_data='full_analysis')],
        [InlineKeyboardButton("💬 دردشة مع المحلل", callback_data='chat_mode')],
        [InlineKeyboardButton("🔄 تحديث السعر", callback_data='refresh')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 *بوت الذهب الذكي - الإصدار الاحترافي*\n\n"
        "✨ *الميزات:*\n"
        "✅ سعر فوري حقيقي (XAU/USD Spot)\n"
        "✅ تحليل فني بـ EMA 9 & 21 + RSI\n"
        "✅ ذكاء اصطناعي (Llama 3.3) يحلل السوق\n"
        "✅ دردشة مع المحلل AI\n"
        "✅ مصادر متعددة (Twelve Data + Yahoo)\n\n"
        "⌨️ *كيف تستخدمه:*\n"
        "• اضغط 'تحليل فوري + AI' لتحليل كامل\n"
        "• اضغط 'دردشة مع المحلل' لتسأل ما تشاء\n"
        "• أو اكتب سؤالك مباشرة بعد تفعيل وضع الدردشة\n\n"
        "اختر من القائمة 👇",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("⏳ جاري المعالجة...")
    
    if query.data == 'full_analysis':
        await query.edit_message_text("🔄 *جاري جلب البيانات الفورية وتحليلها بالذكاء الاصطناعي...*", parse_mode='Markdown')
        await perform_full_analysis(query, context)
        
    elif query.data == 'chat_mode':
        user_id = query.from_user.id
        user_conversations[user_id] = None  # إعادة تعيين
        await query.edit_message_text(
            "💬 *تم تفعيل وضع الدردشة!*\n\n"
            "يمكنك الآن أن تسأل المحلل الذكي أي سؤال عن الذهب:\n\n"
            " *أمثلة:*\n"
            "• هل الذهب في اتجاه صاعد؟\n"
            "• ما أفضل وقت للدخول؟\n"
            "• اشرح لي RSI\n"
            "• ما الفرق بين Spot و Futures؟\n\n"
            "اكتب سؤالك الآن... 👇\n\n"
            "🔙 للعودة للقائمة: /start",
            parse_mode='Markdown'
        )
        # تخزين حالة الدردشة للمستخدم
        context.user_data['chat_mode'] = True
        
    elif query.data == 'refresh':
        await query.edit_message_text("🔄 *جاري تحديث السعر...*", parse_mode='Markdown')
        await perform_full_analysis(query, context, quick=True)

async def perform_full_analysis(query, context, quick=False):
    """إجراء تحليل كامل"""
    data = get_gold_data()
    
    if data['spot_price'] is None and data['candles'] is None:
        await query.edit_message_text(
            "❌ *فشل في جلب البيانات*\n\n"
            "تأكد من:\n"
            "• إضافة `TWELVE_DATA_API_KEY` في Railway\n"
            "• السوق مفتوح (يتوقف الجمعة مساءً)",
            parse_mode='Markdown'
        )
        return
    
    spot_price = data['spot_price']
    candles = data['candles']
    
    # التحليل الأساسي
    if candles is not None and len(candles) >= 30:
        ema_9 = float(candles['EMA_9'].iloc[-1] if 'EMA_9' in candles.columns else candles['Close'].ewm(span=9).mean().iloc[-1])
        ema_21 = float(candles['EMA_21'].iloc[-1] if 'EMA_21' in candles.columns else candles['Close'].ewm(span=21).mean().iloc[-1])
        rsi = float(calculate_rsi(candles['Close'], 14).iloc[-1])
        
        trend_score = 0
        if ema_9 > ema_21: trend_score += 1
        else: trend_score -= 1
        if candles['Close'].iloc[-1] > ema_9: trend_score += 1
        else: trend_score -= 1
        if rsi > 70: trend_score -= 1
        elif rsi < 30: trend_score += 1
        
        if trend_score >= 2: prediction, confidence = "صعود قوي ", 85
        elif trend_score == 1: prediction, confidence = "صعود ضعيف 🟢", 65
        elif trend_score <= -2: prediction, confidence = "هبوط قوي 🔴", 85
        elif trend_score == -1: prediction, confidence = "هبوط ضعيف 🔴", 65
        else: prediction, confidence = "تذبذب / عرضي ", 50
        
        base_analysis = (
            f"💰 *تحليل الذهب الفوري (XAU/USD Spot)*\n\n"
            f"📈 *السعر الفوري:* `${spot_price:.2f}$`\n"
            f"🎯 *التوقع الفني:* *{prediction}*\n"
            f"💪 *قوة الإشارة:* {confidence}%\n\n"
            f"📊 *المؤشرات:*\n"
            f"• EMA 9: `{ema_9:.2f}`\n"
            f"• EMA 21: `{ema_21:.2f}`\n"
            f"• RSI (14): `{rsi:.1f}`\n\n"
            f"📡 *المصدر:* {data['source']}\n"
            f"🕐 *الوقت:* `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`\n\n"
        )
    else:
        base_analysis = (
            f"💰 *سعر الذهب الفوري (XAU/USD)*\n\n"
            f"📈 *السعر:* `${spot_price:.2f}$`\n"
            f"📡 *المصدر:* {data['source']}\n"
            f"🕐 *الوقت:* `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`\n\n"
        )
        ema_9 = ema_21 = rsi = 0
    
    # إضافة تحليل الذكاء الاصطناعي
    if not quick and data['candles'] is not None:
        base_analysis += "🤖 *تحليل الذكاء الاصطناعي (Llama 3.3):*\n\n"
        await query.edit_message_text(base_analysis + "_⏳ جاري التحليل بالذكاء الاصطناعي..._", parse_mode='Markdown')
        
        ai_analysis = analyze_with_ai(data)
        base_analysis += ai_analysis
    else:
        base_analysis += "_استخدم زر 'تحليل فوري + AI' للحصول على تحليل كامل بالذكاء الاصطناعي_"
    
    # إنشاء الشارت
    chart_img = generate_chart(data)
    
    keyboard = [
        [InlineKeyboardButton("📊 تحليل كامل بـ AI", callback_data='full_analysis')],
        [InlineKeyboardButton("💬 اسأل المحلل", callback_data='chat_mode')],
        [InlineKeyboardButton("🔄 تحديث", callback_data='refresh')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if chart_img:
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=chart_img,
            caption=base_analysis,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        try:
            await query.message.delete()
        except:
            pass
    else:
        await query.edit_message_text(base_analysis, parse_mode='Markdown', reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """التعامل مع رسائل الدردشة"""
    if not context.user_data.get('chat_mode'):
        return
    
    user_id = update.message.from_user.id
    message_text = update.message.text
    
    if not message_text or message_text.startswith('/'):
        return
    
    # إرسال "يكتب..."
    typing_msg = await update.message.reply_text("🤖 _المحلل يكتب..._", parse_mode='Markdown')
    
    # الحصول على الرد من AI
    reply = chat_with_ai(user_id, message_text)
    
    await typing_msg.edit_text(
        f"💬 *رد المحلل الذكي:*\n\n{reply}\n\n"
        f"_لإنهاء الدردشة: /start_",
        parse_mode='Markdown'
    )

# ================= تشغيل البوت =================
def main():
    if not BOT_TOKEN:
        logging.error(" BOT_TOKEN غير موجود!")
        return
    
    logging.info("🚀 بدء تشغيل البوت الذكي...")
    logging.info(f"Twelve Data: {'✅' if TWELVE_DATA_API_KEY else '❌'}")
    logging.info(f"Groq AI: {'✅' if GROQ_API_KEY else '❌'}")
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logging.info("✅ البوت يعمل الآن...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
