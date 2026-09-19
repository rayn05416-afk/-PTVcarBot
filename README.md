# PTVcarBot

Telegram bot for collecting vehicle/report photos during the workday and sorting them into PDFs.

## Workflow
1. Send paper report, vehicle/tow photo, and app screenshot in any order.
2. Press **فرز اليوم**.
3. The bot tries to anchor each group on the **vehicle plate from the paper report**, and uses the **8-digit report number from the app** as the PDF report number.
4. PDF filename: `Arabic plate letters + English digits - report number.pdf`
5. Uncertain images are not guessed; they are reported as needing review.

## Render
- Build: `pip install -r requirements.txt`
- Start: `uvicorn bot_web:app --host 0.0.0.0 --port $PORT`
- Environment variables:
  - `BOT_TOKEN` = BotFather token
  - `WEBHOOK_SECRET` = any long random secret
- Render automatically provides `RENDER_EXTERNAL_URL`; the app uses it to register the Telegram webhook.

For real work records, use an approved paid storage/hosting setup. Render Free has an ephemeral filesystem, so files can be lost when the service restarts or spins down.
