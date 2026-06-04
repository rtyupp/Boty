import os
import io
import logging
import yfinance as yf
import mplfinance as mpf
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

# ================= إعدادات البوت =================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
SYMBOL = "GC=F"  # الذهب

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ================= دوال جلب البيانات والتحليل =================
def get_gold_data():
    """جلب بيانات الذهب لفريم 5 دقائق (آخر 3 أيام)"""
    try:
        # ملاحظة: yfinance يدعم فريم 5 دقائق لمدة 60 يوم كحد أقصى
        data = yf.download(SYMBOL, period='5d', interval='5m', progress=False)
        if data.empty:
            logging.warning("لا توجد بيانات من Yahoo Finance")
            return None
        return data
    except Exception as e:
        logging.error(f"خطأ في جلب البيانات: {e}")
        return None

def analyze_trend(data):
    """تحليل الاتجاه وإعطاء توقع"""
    if data is None or len(data) < 50:
        return "⚠️ بيانات غير كافية للتحليل", 0, 0, 0

    # حساب المؤشرات المناسبة لفريم 5 دقائق
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
    last_price = float(data['Close'].iloc[-1])
    ema_9 = float(data['EMA_9'].iloc[-1])
    ema_21 = float(data['EMA_21'].iloc[-1])
    
    # تحليل بسيط
    if ema_9 > ema_21 and last_price > ema_9:
        prediction = "صعود قوي 🟢"
        confidence = 85
    elif ema_9 > ema_21:
        prediction = "صعود ضعيف 🟢"
        confidence = 65
    elif ema_9 < ema_21 and last_price < ema_9:
        prediction = "هبوط قوي 🔴"
        confidence = 85
    elif ema_9 < ema_21:
        prediction = "هبوط ضعيف 🔴"
        confidence = 65
    else:
        prediction = "تذبذب / عرضي "
        confidence = 50
    
    return prediction, confidence, ema_9, ema_21

def generate_chart(data, prediction):
    """إنشاء الشارت وإرجاعه كصورة في الذاكرة"""
    if data is None:
        return None
    
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
    # تلوين المؤشرات
    ema9_color = '#00FF00' if 'صعود' in prediction else '#FF0000'
    ema21_color = '#FFA500'
    
    add_plots = [
        mpf.make_addplot(data['EMA_9'], color=ema9_color, width=1.5, label='EMA 9'),
        mpf.make_addplot(data['EMA_21'], color=ema21_color, width=1.5, label='EMA 21')
    ]
    
    # حفظ الصورة في الذاكرة (بدون كتابة على الخادم)
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
        "✅ توقعات صعود/هبوط\n"
        "✅ شارت مباشر مع EMA 9 & 21\n\n"
        "اضغط الزر أدناه للبدء 👇",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("⏳ جاري التحليل...")
    
    await query.edit_message_text(" *جاري جلب بيانات الذهب وتحليلها...*", parse_mode='Markdown')
    
    # جلب البيانات
    data = get_gold_data()
    if data is None:
        await query.edit_message_text("❌ *فشل في جلب البيانات. حاول مرة أخرى بعد دقيقة.*", parse_mode='Markdown')
        return
    
    # التحليل
    prediction, confidence, ema_9, ema_21 = analyze_trend(data)
    last_price = float(data['Close'].iloc[-1])
    
    # إنشاء الشارت
    chart_img = generate_chart(data, prediction)
    
    # بناء الرسالة
    message_text = (
        f"💰 *تحليل الذهب (XAU/USD)*\n"
        f" *الفريم:* 5 دقائق\n\n"
        f"📈 *السعر الحالي:* `${last_price:.2f}$`\n"
        f"🎯 *التوقع:* *{prediction}*\n"
        f" *قوة الإشارة:* {confidence}%\n\n"
        f"📊 *المؤشرات:*\n"
        f"• EMA 9: `{ema_9:.2f}`\n"
        f"• EMA 21: `{ema_21:.2f}`\n\n"
        f"⚡ _الفريم 5 دقائق سريع - استخدم للتداول قصير المدى (Scalping)_",
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
        # حذف رسالة "جاري التحليل" القديمة
        try:
            await query.message.delete()
        except:
            pass
    else:
        await query.edit_message_text("❌ *حدث خطأ في إنشاء الشارت.*", parse_mode='Markdown', reply_markup=reply_markup)

# ================= تشغيل البوت =================
def main():
    if not BOT_TOKEN:
        logging.error("BOT_TOKEN غير موجود!")
        return
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_button))
    
    logging.info("🚀 البوت يعمل الآن...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == '__main__':
    main()
