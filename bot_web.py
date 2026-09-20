import os
import re
import shutil
import asyncio
import logging
from pathlib import Path
from datetime import datetime

import cv2
import pytesseract
from ultralytics import YOLO
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

# Render provides this automatically for Web 
PUBLIC_URL = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").rstrip("/")
if PUBLIC_URL and not PUBLIC_URL.startswith("http"):
    PUBLIC_URL = "https://" + PUBLIC_URL
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "ptvcar-webhook-secret")
PORT = int(os.getenv("PORT", "10000"))

DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data"))
# If /var/data is not mounted, local storage still works for testing.
try:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    DATA_ROOT = BASE / "data"
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

OUTPUT_ROOT = DATA_ROOT / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PTVcarBot")

AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PLATE_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

app = FastAPI(title="PTVcarBot")
bot_app = Application.builder().token(TOKEN).build()


def normalize_digits(s: str) -> str:
    return (s or "").translate(AR_DIGITS)


def extract_report_numbers(text: str):
    text = normalize_digits(text)
    return sorted(set(re.findall(r"(?<!\d)\d{8}(?!\d)", text)))


def normalize_plate_arabic(text: str):
    """Return Arabic letters + English digits, e.g. 'ت ع ر 8627'."""
    text = normalize_digits(text or "")
    # OCR can put spaces/punctuation between characters.
    compact = re.sub(r"[^\u0600-\u06FF0-9]", "", text)

    patterns = [
        r"([ابتثجحخدذرزسشصضطظعغفقكلمنهوي]{1,3})(\d{3,4})",
        r"(\d{3,4})([ابتثجحخدذرزسشصضطظعغفقكلمنهوي]{1,3})",
    ]
    for p in patterns:
        m = re.search(p, compact)
        if m:
            a, b = m.group(1), m.group(2)
            if a.isdigit():
                digits, letters = a, b
            else:
                letters, digits = a, b
            # Saudi plate filename convention requested by user.
            return f"{' '.join(letters)} {digits}"

    # Fallback for spaced OCR such as ت ع ر 8627.
    letters = re.findall(r"[ابتثجحخدذرزسشصضطظعغفقكلمنهوي]", text)
    nums = re.findall(r"(?<!\d)\d{3,4}(?!\d)", text)
    if letters and nums:
        # Prefer 1-3 letters immediately near a number.
        for n in nums:
            pos = text.find(n)
            nearby = text[max(0, pos - 12): pos + len(n) + 3]
            ls = re.findall(r"[ابتثجحخدذرزسشصضطظعغفقكلمنهوي]", nearby)
            if 1 <= len(ls) <= 3:
                return f"{' '.join(ls)} {n}"
    return None


def plate_key(plate: str):
    if not plate:
        return None
    return re.sub(r"\s+", "", plate)


def ocr_image(path: Path):

    img = cv2.imread(str(path))

    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # تكبير الصورة
    gray = cv2.resize(
        gray,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    # تحسين الصورة
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    texts = []

    for candidate in (gray, th):

        for psm in (6, 7, 11, 12):

            try:

                result = pytesseract.image_to_string(
                    candidate,
                    lang="ara+eng",
                    config=f"--psm {psm}"
                )

                if result:
                    texts.append(result)

            except Exception as exc:

                logger.warning(
                    "OCR failed: %s",
                    exc
                )

    return "\n".join(texts)


def vehicle_plate_ocr(path: Path):

    img = cv2.imread(str(path))

    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # تكبير الصورة
    gray = cv2.resize(
        gray,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    # استخراج الحواف
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if h <= 0:
            continue

        ratio = w / float(h)
        area = w * h

        # لوحة السيارة غالبًا مستطيلة
        if 1.5 <= ratio <= 6.0 and area > 1500:

            candidates.append(
                (area, x, y, w, h)
            )

    # الأكبر أولاً
    candidates.sort(
        reverse=True
    )

    texts = []

    # نجرب أفضل المناطق المرشحة
    for _, x, y, w, h in candidates[:15]:

        crop = gray[
            y:y + h,
            x:x + w
        ]

        if crop.size == 0:
            continue

        _, threshold = cv2.threshold(
            crop,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        for candidate in (
            crop,
            threshold
        ):

            try:

                result = pytesseract.image_to_string(
                    candidate,
                    lang="ara+eng",
                    config="--psm 7"
                )

                if result:
                    texts.append(result)

            except Exception as exc:

                logger.warning(
                    "Vehicle plate OCR failed: %s",
                    exc
                )

    return "\n".join(texts)
    img = cv2.imread(str(path))

    if img is None:
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # تكبير الصورة لتحسين قراءة النص واللوحة
    gray = cv2.resize(
        gray,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    # تحسين التباين
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # أكثر من نسخة للصورة
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    variants = [
        gray,
        otsu,
        adaptive
    ]

    texts = []

    for candidate in variants:
        for psm in (6, 11, 12):
            try:
                result = pytesseract.image_to_string(
                    candidate,
                    lang="ara+eng",
                    config=f"--psm {psm}"
                )

                if result:
                    texts.append(result)

            except Exception as exc:
                logger.warning("OCR failed: %s", exc)

    return "\n".join(texts)
    img = cv2.imread(str(path))
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texts = []
    for candidate in (gray, th):
        for psm in (6, 11):
            try:
                texts.append(pytesseract.image_to_string(candidate, lang="ara+eng", config=f"--psm {psm}"))
            except Exception as exc:
                logger.warning("OCR failed: %s", exc)
    return "\n".join(texts)

def classify_image(text: str):
    text = text or ""

    # صورة التطبيق
    app_words = (
        "نوع المحضر",
        "حالة المحضر",
        "المخالفات",
        "حالة الطلب",
        "الإجراء",
        "التطبيق"
    )

    if "رقم المحضر" in text and any(
        word in text for word in app_words
    ):
        return "app"

    # المحضر الورقي
    paper_words = (
        "اسم مستلم",
        "رقم هوية المخالف",
        "المحضر الورقي",
        "بيانات المركبة",
        "لوحة المركبة",
        "نوع المركبة"
    )

    if any(word in text for word in paper_words):
        return "paper"

    # إذا وجد رقم محضر ولا توجد علامات واضحة للتطبيق،
    # نعتبرها مرشحًا للمحضر.
    if extract_report_numbers(text):
        return "paper"

    return "photo"


def user_folder(user_id: int) -> Path:
    p = DATA_ROOT / "users" / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_folder(user_id: int) -> Path:
    p = OUTPUT_ROOT / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def image_pdf_page(c, path: Path):
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    W, H = A4
    margin = 18
    scale = min((W - 2 * margin) / iw, (H - 2 * margin) / ih)
    dw, dh = iw * scale, ih * scale
    x, y = (W - dw) / 2, (H - dh) / 2
    c.drawImage(ImageReader(im), x, y, width=dw, height=dh, preserveAspectRatio=True)
    c.showPage()


def make_pdf(user_id: int, report: str, plate: str, paths):
    safe_plate = re.sub(r'[\\/:*?"<>|]', "_", plate)
    out = output_folder(user_id) / f"{safe_plate} - {report}.pdf"
    c = canvas.Canvas(str(out), pagesize=A4)
    for p in paths:
        image_pdf_page(c, p)
    c.save()
    return out


def message_target(update: Update):
    if update.message:
        return update.message
    if update.callback_query and update.callback_query.message:
        return update.callback_query.message
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[
        InlineKeyboardButton("📦 فرز اليوم", callback_data="finish"),
        InlineKeyboardButton("🗑️ مسح اليوم", callback_data="reset"),
    ]]
    target = message_target(update)
    if target:
        await target.reply_text(
            "أهلاً بك في PTVcarBot 🚗\n\n"
            "أرسل صور المحاضر والسطحات وصور التطبيق بأي ترتيب.\n"
            "وعند الانتهاء اضغط «فرز اليوم».\n\n"
            "الربط يعتمد على لوحة المركبة من المحضر، ورقم المحضر من التطبيق.\n"
            "أي صورة غير مؤكدة ستذهب إلى «تحتاج مراجعة» بدل التخمين.",
            reply_markup=InlineKeyboardMarkup(kb),
        )


async def save_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    folder = user_folder(uid)
    photo = update.message.photo[-1]
    tg_file = await context.bot.get_file(photo.file_id)
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{photo.file_unique_id}.jpg"
    path = folder / name
    await tg_file.download_to_drive(custom_path=str(path))
    await update.message.reply_text("✅ تم حفظ الصورة. أرسل الباقي بأي ترتيب.")


async def finish(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    folder = user_folder(uid)

    paths = sorted(folder.glob("*.jpg"))

    target = message_target(update)

    if not paths:
        if target:
            await target.reply_text("لا توجد صور لهذا اليوم.")
        return

    if target:
        await target.reply_text(
            "🔎 أفحص المحاضر وأقرأ اللوحات وأرقام المحاضر..."
        )

    records = []

    for p in paths:
        text = await asyncio.to_thread(ocr_image, p)

        reports = extract_report_numbers(text)
        plate = normalize_plate_arabic(text)
        kind = classify_image(text)

        records.append({
            "path": p,
            "text": text,
            "reports": reports,
            "plate": plate,
            "kind": kind
        })

    created = []
    review = []
    used = set()

    # -----------------------------------------
    # 1) المحاضر هي نقطة البداية
    # -----------------------------------------

    papers = [
        r for r in records
        if r["kind"] == "paper"
    ]

    apps = [
        r for r in records
        if r["kind"] == "app"
    ]

    photos = [
        r for r in records
        if r["kind"] == "photo"
    ]

    # -----------------------------------------
    # 2) معالجة كل محضر
    # -----------------------------------------

    for paper in papers:

        plate = paper["plate"]

        if not plate:
            review.append(paper)
            continue

        plate_k = plate_key(plate)

        # رقم المحضر من الورقي إن وجد
        report_numbers = list(paper["reports"])

        # -----------------------------------------
        # 3) البحث عن التطبيق بواسطة رقم المحضر
        # -----------------------------------------

        matching_apps = []

        for app_item in apps:

            if str(app_item["path"]) in used:
                continue

            if not report_numbers:
                continue

            if any(
                number in app_item["reports"]
                for number in report_numbers
            ):
                matching_apps.append(app_item)

        # -----------------------------------------
        # 4) إذا لم يوجد رقم في الورقي
        # نبحث عن تطبيق يحتوي لوحة المركبة
        # -----------------------------------------

        if not report_numbers:

            for app_item in apps:

                if str(app_item["path"]) in used:
                    continue

                if plate_key(app_item["plate"]) == plate_k:

                    report_numbers.extend(
                        app_item["reports"]
                    )

                    if report_numbers:
                        matching_apps.append(app_item)
                        break

        # يجب أن يكون لدينا رقم محضر واحد
        report_numbers = sorted(set(report_numbers))

        if len(report_numbers) != 1:
            review.append(paper)
            continue

        report = report_numbers[0]

        # -----------------------------------------
        # 5) البحث عن صورة المركبة بواسطة اللوحة
        # -----------------------------------------

        matching_photos = []

        for photo in photos:

            if str(photo["path"]) in used:
                continue

            if plate_key(photo["plate"]) == plate_k:
                matching_photos.append(photo)

        # -----------------------------------------
        # 6) تجهيز الصور
        # المحضر -> المركبة -> التطبيق
        # -----------------------------------------

        group = [paper]

        group.extend(matching_photos)

        group.extend(matching_apps)

        # -----------------------------------------
        # 7) إنشاء PDF
        # -----------------------------------------

        pdf = make_pdf(
            uid,
            report,
            plate,
            [x["path"] for x in group]
        )

        created.append(pdf)

        # تسجيل الصور المستخدمة
        for item in group:
            used.add(str(item["path"]))

    # -----------------------------------------
    # 8) أي صورة لم تستخدم تذهب للمراجعة
    # -----------------------------------------

    for record in records:

        path_key = str(record["path"])

        if path_key not in used:
            review.append(record)

    # إزالة التكرار
    unique_review = []
    review_paths = set()

    for item in review:

        key = str(item["path"])

        if key not in review_paths:
            review_paths.add(key)
            unique_review.append(item)

    # -----------------------------------------
    # 9) إرسال ملفات PDF
    # -----------------------------------------

    for pdf in created:

        with open(pdf, "rb") as f:

            if target:
                await target.reply_document(
                    document=f,
                    filename=pdf.name
                )

    if created and target:

        await target.reply_text(
            f"✅ تم إنشاء {len(created)} ملف PDF."
        )

    # -----------------------------------------
    # 10) الصور غير المطابقة
    # -----------------------------------------

    if unique_review and target:

        await target.reply_text(
            f"⚠️ {len(unique_review)} صورة تحتاج مراجعة.\n\n"
            "لم أقم بالتخمين في اللوحة أو رقم المحضر."
        )

    # -----------------------------------------
    # 11) تنظيف صور اليوم بعد انتهاء الفرز
    # -----------------------------------------

    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    folder = user_folder(uid)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    target = message_target(update)
    if target:
        await target.reply_text("🗑️ تم مسح صور اليوم.")


async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    if update.callback_query.data == "finish":
        await finish(update, context)
    elif update.callback_query.data == "reset":
        await reset(update, context)


bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(CommandHandler("finish", finish))
bot_app.add_handler(CommandHandler("reset", reset))
bot_app.add_handler(CallbackQueryHandler(buttons))
bot_app.add_handler(MessageHandler(filters.PHOTO, save_photo))


@app.get("/")
async def health():
    return {"status": "ok", "bot": "PTVcarBot"}


@app.post(f"/telegram/webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        update = Update.de_json(data, bot_app.bot)
        # Telegram only needs a fast 2xx response; PTB processes the update
        # asynchronously so image downloads/OCR don't block Telegram.
        asyncio.create_task(bot_app.process_update(update))
        return JSONResponse({"ok": True})
    except Exception as exc:
        logger.exception("Webhook error: %s", exc)
        raise HTTPException(status_code=400, detail="invalid update")


@app.on_event("startup")
async def startup():
    await bot_app.initialize()
    await bot_app.start()
    if not PUBLIC_URL:
        logger.warning("RENDER_EXTERNAL_URL is missing; webhook was not set")
        return
    webhook_url = f"{PUBLIC_URL}/telegram/webhook/{WEBHOOK_SECRET}"
    await bot_app.bot.set_webhook(url=webhook_url, secret_token=WEBHOOK_SECRET, drop_pending_updates=False)
    logger.info("Webhook set: %s", webhook_url)


@app.on_event("shutdown")
async def shutdown():
    try:
        await bot_app.bot.delete_webhook(drop_pending_updates=False)
    finally:
        await bot_app.stop()
        await bot_app.shutdown()
