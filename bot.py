import os
import re
import shutil
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

import cv2
import pytesseract
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')
DATA = BASE / "data"
OUTPUT = BASE / "output"
DATA.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise SystemExit("ضع BOT_TOKEN في متغيرات البيئة أو ملف .env قبل التشغيل.")

# Arabic -> English digit normalization
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Arabic plate letters commonly represented on Saudi plates.
# The actual filename uses Arabic letters, exactly as requested.
PLATE_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

def normalize_digits(s):
    return s.translate(AR_DIGITS)

def clean_text(s):
    s = normalize_digits(s or "")
    s = re.sub(r"[^\w\u0600-\u06FF\s-]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

def extract_report_numbers(text):
    text = normalize_digits(text)
    # Prefer 8-digit numbers, typical for the app reference.
    return sorted(set(re.findall(r"\b\d{8}\b", text)))

def normalize_plate_arabic(text):
    """
    Try to find a Saudi-style plate representation in OCR text.
    Output convention: Arabic letters separated by spaces + English digits.
    This is deliberately conservative; ambiguous cases go to review.
    """
    text = normalize_digits(text)
    # Look for a run containing Arabic letters and 3-4 digits.
    patterns = [
        r"([ابتثجحخدذرزسشصضطظعغفقكلمنهوي]{1,3})\s*[- ]?\s*(\d{3,4})",
        r"(\d{3,4})\s*[- ]?\s*([ابتثجحخدذرزسشصضطظعغفقكلمنهوي]{1,3})",
    ]
    for p in patterns:
        m = re.search(p, text)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if a.isdigit():
            digits, letters = a, b
        else:
            letters, digits = a, b
        letters = " ".join(list(letters))
        return f"{letters} {digits}"
    return None

def ocr_image(path):
    """
    OCR in Arabic + English. For difficult photos, create an enlarged,
    contrast-enhanced version before OCR.
    """
    img = cv2.imread(str(path))
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 2.0
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    texts = []
    for candidate in (gray, th):
        try:
            texts.append(pytesseract.image_to_string(candidate, lang="ara+eng", config="--psm 6"))
        except Exception:
            texts.append("")
    return "\n".join(texts)

def classify_image(text):
    low = text.lower()
    # App screenshots tend to contain these labels.
    if "رقم المحضر" in text and ("نوع المحضر" in text or "المخالفات" in text):
        return "app"
    if "رقم المحضر" in text and ("اسم مستلم" in text or "رقم هوية المخالف" in text):
        return "paper"
    return "photo"

def image_pdf_page(c, path):
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    W, H = A4
    margin = 18
    scale = min((W-2*margin)/iw, (H-2*margin)/ih)
    dw, dh = iw*scale, ih*scale
    x, y = (W-dw)/2, (H-dh)/2
    c.drawImage(ImageReader(im), x, y, width=dw, height=dh,
                preserveAspectRatio=True)
    c.showPage()

def make_pdf(report, plate, paths):
    safe_plate = re.sub(r'[\\/:*?"<>|]', "_", plate)
    out = OUTPUT / f"{safe_plate} - {report}.pdf"
    c = canvas.Canvas(str(out), pagesize=A4)
    for p in paths:
        image_pdf_page(c, p)
    c.save()
    return out

def user_folder(user_id):
    p = DATA / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton("📦 فرز اليوم", callback_data="finish"),
           InlineKeyboardButton("🗑️ مسح اليوم", callback_data="reset")]]
    await update.message.reply_text(
        "أهلاً بك في PTVcarBot\n\n"
        "أرسل صور المحاضر والسطحات وصور التطبيق بأي ترتيب.\n"
        "وعند الانتهاء اضغط «فرز اليوم».",
        reply_markup=InlineKeyboardMarkup(kb)
    )

async def save_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    folder = user_folder(uid)
    photos = update.message.photo
    if not photos:
        return
    photo = photos[-1]
    tg_file = await context.bot.get_file(photo.file_id)
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{photo.file_unique_id}.jpg"
    path = folder / name
    await tg_file.download_to_drive(custom_path=str(path))
    await update.message.reply_text("✅ تم حفظ الصورة. أرسل الباقي بأي ترتيب.")

async def finish(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    folder = user_folder(uid)
    paths = sorted(folder.glob("*"))
    if not paths:
        await update.message.reply_text("لا توجد صور لهذا اليوم.")
        return

    await update.message.reply_text("🔎 أفحص الصور وأطابق رقم المحضر واللوحة...")
    records = []
    for p in paths:
        text = await asyncio.to_thread(ocr_image, p)
        reports = extract_report_numbers(text)
        plate = normalize_plate_arabic(text)
        kind = classify_image(text)
        records.append({"path": p, "text": text, "reports": reports, "plate": plate, "kind": kind})

    # Group by report number. If an image has no report number, it is kept for
    # later matching/review rather than guessed into a report.
    groups = {}
    review = []
    for r in records:
        if not r["reports"]:
            review.append(r)
            continue
        for rep in r["reports"]:
            groups.setdefault(rep, []).append(r)

    created = []
    for report, items in groups.items():
        # Determine the Arabic plate from paper/app OCR.
        plate = next((x["plate"] for x in items if x["kind"] == "paper" and x["plate"]), None)
        if not plate:
            plate = next((x["plate"] for x in items if x["plate"]), None)

        if not plate:
            review.extend(items)
            continue

        # Keep each image once. We intentionally do not use the tow truck's
        # plate as the primary key; the paper/app report is the anchor.
        unique = []
        seen = set()
        for x in items:
            key = str(x["path"])
            if key not in seen:
                seen.add(key)
                unique.append(x)

        # Prefer paper -> tow/photo -> app ordering.
        order = {"paper": 0, "photo": 1, "app": 2}
        unique.sort(key=lambda x: order.get(x["kind"], 1))
        pdf = make_pdf(report, plate, [x["path"] for x in unique])
        created.append(pdf)

    if created:
        for pdf in created:
            with open(pdf, "rb") as f:
                await update.message.reply_document(document=f, filename=pdf.name)
        await update.message.reply_text(f"✅ تم إنشاء {len(created)} ملف PDF.")
    else:
        await update.message.reply_text(
            "⚠️ لم أستطع تأكيد أي مجموعة بثقة.\n"
            "هذا طبيعي في الصور ذات الكتابة اليدوية؛ لا أريد تخمين لوحة أو محضر بشكل خاطئ."
        )

    # Clear today's queue after processing.
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    folder = user_folder(uid)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    await update.message.reply_text("🗑️ تم مسح صور اليوم.")

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "finish":
        await finish(update, context)
    elif q.data == "reset":
        await reset(update, context)

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("finish", finish))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CallbackQueryHandler(buttons))
    app.add_handler(MessageHandler(filters.PHOTO, save_photo))
    print("PTVcarBot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
