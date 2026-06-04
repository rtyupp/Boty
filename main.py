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
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# ================= الإعدادات =================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# تخزين محادثات المستخدمين
user_chats = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json',
}

# ================= 1. جلب السعر الفوري =================

def get_spot_price():
    """جلب السعر الفوري من Twelve Data"""
    if not TWELVE_DATA_API_KEY:
        logging.warning("TWELVE_DATA_API_KEY غير موجود")
        return None
    
    try:
        url = "https://api.twelvedata.com/price"
        params = {'symbol': 'XAU/USD', 'apikey': TWELVE_DATA_API_KEY}
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'price' in data:
            price = float(data['price'])
            logging.info(f"✅ السعر الفوري: ${price:.2f}")
            return price
        
        logging.warning(f"Twelve Data response: {data}")
        return None
        
    except Exception as e:
        logging.error(f"Spot price error: {e}")
        return None

# ================= 2. جلب بيانات الشموع =================

def get_candlestick_data(interval='5min', outputsize=50):
    """جلب بيانات الشموع"""
    if not TWELVE_DATA_API_KEY:
        logging.warning("TWELVE_DATA_API_KEY غير موجود")
        return None
    
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
        
        if 'values' not in data:
            error_msg = data.get('message', 'بيانات غير متاحة')
            logging.error(f"API Error: {error_msg}")
            return None
        
        # تحويل لـ DataFrame
        df = pd.DataFrame(data['values'])
        
        # تحويل الأنواع الرقمية
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)  # ترتيب من الأقدم للأحدث
        
        df.rename(columns={
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        df.dropna(inplace=True)
        
        if len(df) < 30:
            logging.warning(f"بيانات قليلة: {len(df)} شمعة فقط")
            return None
        
        logging.info(f"✅ تم جلب {len(df)} شمعة")
        return df
        
    except Exception as e:
        logging.error(f"Candlestick error: {e}")
        return None

# ================= 3. حساب المؤشرات =================

def calculate_indicators(df):
    """حساب EMA و RSI"""
    try:
        df = df.copy()
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # حساب RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df.dropna(inplace=True)
        return df
        
    except Exception as e:
        logging.error(f"Indicators error: {e}")
        return None

# ================= 4. إنشاء الشارت =================

def create_chart(df):
    """إنشاء صورة الشارت"""
    if df is None or len(df) < 30:
        return None
    
    try:
        add_plots = [
            mpf.make_addplot(df['EMA_9'], color='#00FF00', width=1.5),
            mpf.make_addplot(df['EMA_21'], color='#FFA500', width=1.5)
        ]
        
        img_buffer = io.BytesIO()
        
        mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':')
        
        mpf.plot(
            df, type='candle', addplot=add_plots,
            title='Gold (XAU/USD) - 5min',
            style=s, figsize=(12, 7), savefig=img_buffer,
            tight_layout=True
        )
        
        img_buffer.seek(0)
        logging.info("✅ تم إنشاء الشارت")
        return img_buffer
        
    except Exception as e:
        logging.error(f"Chart error: {e}")
        return None

# ================= 5. التحليل الفني البسيط =================

def simple_analysis(df):
    """تحليل فني بسيط بدون AI"""
    if df is None or len(df) < 30:
        return None
    
    try:
        last_close = float(df['Close'].iloc[-1])
        ema_9 = float(df['EMA_9'].iloc[-1])
        ema_21 = float(df['EMA_21'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        
        # حساب الاتجاه
        score = 0
        if ema_9 > ema_21:
            score += 1
        else:
            score -= 1
        
        if last_close > ema_9:
            score += 1
        else:
            score -= 1
        
        if rsi > 70:
            score -= 1  # تشبع شرائي
        elif rsi < 30:
            score += 1  # تشبع بيعي
        
        # تحديد التوقع
        if score >= 2:
            prediction = "صعود قوي 🟢"
            confidence = 85
        elif score == 1:
            prediction = "صعود ضعيف 🟢"
            confidence = 65
        elif score <= -2:
            prediction = "هبوط قوي 🔴"
            confidence = 85
        elif score == -1:
            prediction = "هبوط ضعيف 🔴"
            confidence = 65
        else:
            prediction = "تذبذب / عرضي 🟡"
            confidence = 50
        
        return {
            'price': last_close,
            'ema_9': ema_9,
            'ema_21': ema_21,
            'rsi': rsi,
            'prediction': prediction,
            'confidence': confidence
        }
        
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return None

# ================= 6. الذكاء الاصطناعي =================

def ai_analysis(spot_price, df):
    """تحليل AI باستخدام Groq"""
    if not GROQ_API_KEY:
        return None
    
    if df is None or len(df) < 30:
        return None
    
    try:
        last_10 = df.tail(10)
        
        prompt = f"""أنت محلل فني محترف للذهب (XAU/USD).

السعر الفوري الحالي: ${spot_price:.2f}

آخر 10 شموع (فريم 5 دقائق):
{last_10[['Open', 'High', 'Low', 'Close', 'Volume']].round(2).to_string()}

المؤشرات الأخيرة:
- EMA 9: {df['EMA_9'].iloc[-1]:.2f}
- EMA 21: {df['EMA_21'].iloc[-1]:.2f}
- RSI (14): {df['RSI'].iloc[-1]:.1f}

قدم تحليلاً مختصراً بالعربية يتضمن:
1. الاتجاه العام
2. مناطق الدعم والمقاومة القريبة
3. توصية (شراء/بيع/انتظار)
4. وقف الخسارة وهدف الربح المقترح

كن مختصراً واستخدم emojis."""
        
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "أنت محلل فني خبير في تداول الذهب. ردودك مختصرة بالعربية."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"AI error: {e}")
        return None

# ================= 7. الدردشة مع AI =================

def chat_with_ai(user_id, message):
    """الدردشة مع المحلل الذكي"""
    if not GROQ_API_KEY:
        return "⚠️ خدمة AI غير متاحة. تأكد من إضافة GROQ_API_KEY"
    
    try:
        # تهيئة المحادثة
        if user_id not in user_chats:
            user_chats[user_id] = [
                {"role": "system", "content": "أنت خبير تداول الذهب (XAU/USD). تجيب بالعربية بشكل مختصر ومباشر. تعطي نصائح واقعية وتحذر من المخاطر."}
            ]
        
        user_chats[user_id].append({"role": "user", "content": message})
        
        # الاحتفاظ بآخر 10 رسائل
        if len(user_chats[user_id]) > 11:
            user_chats[user_id] = user_chats[user_id][:1] + user_chats[user_id][-10:]
        
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=user_chats[user_id],
            temperature=0.7,
            max_tokens=800
        )
        
        reply = response.choices[0].message.content
        user_chats[user_id].append({"role": "assistant", "content": reply})
        
        return reply
        
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return f"️ خطأ: {str(e)}"

# ================= 8. دوال تيليجرام =================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """رسالة البداية"""
    keyboard = [
        [InlineKeyboardButton("📊 تحليل كامل + AI", callback_data='full')],
        [InlineKeyboardButton("💬 دردشة مع المحلل", callback_data='chat')],
        [InlineKeyboardButton("⚡ سعر فوري فقط", callback_data='quick')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 *بوت الذهب الذكي*\n\n"
        "✨ الميزات:\n"
        "✅ سعر فوري حقيقي (XAU/USD)\n"
        "✅ تحليل فني (EMA + RSI)\n"
        "✅ ذكاء اصطناعي (Llama 3.3)\n"
        "✅ دردشة مع المحلل\n\n"
        "اختر من القائمة 👇",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'full':
        await query.edit_message_text("🔄 جاري التحليل الكامل...")
        await full_analysis(query, context)
        
    elif query.data == 'chat':
        context.user_data['chat_mode'] = True
        await query.edit_message_text(
            "💬 *تم تفعيل وضع الدردشة!*\n\n"
            "اكتب سؤالك الآن:\n"
            "• هل الاتجاه صاعد؟\n"
            "• ما أفضل وقت للدخول؟\n"
            "• اشرح لي RSI\n\n"
            "للخروج: /start",
            parse_mode='Markdown'
        )
        
    elif query.data == 'quick':
        await query.edit_message_text("⚡ جاري جلب السعر...")
        await quick_price(query, context)

async def full_analysis(query, context):
    """تحليل كامل"""
    # جلب البيانات
    spot_price = get_spot_price()
    df = get_candlestick_data('5min', 50)
    
    if spot_price is None and df is None:
        await query.edit_message_text(
            "❌ فشل في جلب البيانات\n\n"
            "تأكد من:\n"
            "• إضافة TWELVE_DATA_API_KEY في Railway\n"
            "• السوق مفتوح (يغلق الجمعة مساءً)"
        )
        return
    
    # حساب المؤشرات
    if df is not None:
        df = calculate_indicators(df)
    
    # التحليل البسيط
    analysis = simple_analysis(df)
    
    # بناء الرسالة
    message = "💰 *تحليل الذهب (XAU/USD)*\n\n"
    
    if spot_price:
        message += f"📈 السعر الفوري: `${spot_price:.2f}$\n"
    
    if analysis:
        message += (
            f"🎯 التوقع: *{analysis['prediction']}*\n"
            f"💪 قوة الإشارة: {analysis['confidence']}%\n\n"
            f" المؤشرات:\n"
            f"• EMA 9: `{analysis['ema_9']:.2f}`\n"
            f"• EMA 21: `{analysis['ema_21']:.2f}`\n"
            f"• RSI: `{analysis['rsi']:.1f}`\n\n"
        )
    
    # تحليل AI
    if df is not None and spot_price:
        message += "🤖 *تحليل الذكاء الاصطناعي:*\n\n"
        await query.edit_message_text(message + "_ جاري التحليل بالـ AI..._", parse_mode='Markdown')
        
        ai_result = ai_analysis(spot_price, df)
        if ai_result:
            message += ai_result + "\n\n"
        else:
            message += "️ AI غير متاح حالياً\n\n"
    
    message += f"🕐 الوقت: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    
    # إنشاء الشارت
    chart_img = create_chart(df)
    
    keyboard = [
        [InlineKeyboardButton("📊 تحليل كامل", callback_data='full')],
        [InlineKeyboardButton("💬 دردشة", callback_data='chat')],
        [InlineKeyboardButton("⚡ سعر سريع", callback_data='quick')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if chart_img:
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=chart_img,
            caption=message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        try:
            await query.message.delete()
        except:
            pass
    else:
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)

async def quick_price(query, context):
    """سعر سريع فقط"""
    spot_price = get_spot_price()
    
    if spot_price is None:
        await query.edit_message_text("❌ فشل في جلب السعر")
        return
    
    message = f"💰 *السعر الفوري للذهب*\n\n📈 ${spot_price:.2f}\n\n🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    
    keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data='quick')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(message, parse_mode='Markdown', reply_markup=reply_markup)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """التعامل مع الدردشة"""
    if not context.user_data.get('chat_mode'):
        return
    
    if not update.message.text:
        return
    
    user_id = update.message.from_user.id
    message = update.message.text
    
    await update.message.reply_text("🤖 _جاري الكتابة..._", parse_mode='Markdown')
    
    reply = chat_with_ai(user_id, message)
    
    await update.message.reply_text(
        f"💬 *رد المحلل:*\n\n{reply}\n\n_للخروج: /start_",
        parse_mode='Markdown'
    )

# ================= تشغيل البوت =================

def main():
    if not BOT_TOKEN:
        logging.error("BOT_TOKEN غير موجود!")
        return
    
    logging.info("🚀 بدء البوت...")
    logging.info(f"Twelve Data: {'✅' if TWELVE_DATA_API_KEY else '❌'}")
    logging.info(f"Groq AI: {'✅' if GROQ_API_KEY else '❌'}")
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logging.info("✅ البوت يعمل الآن!")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
