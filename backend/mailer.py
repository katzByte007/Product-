"""Gmail SMTP sender for alert assignments."""
import logging
import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from backend.db import get_db
from config import ALERTS_DIR

logger = logging.getLogger(__name__)

AUTH_HELP = (
    "Gmail login failed. Use a 16-character App Password with no spaces "
    "(Google Account → Security → 2-Step Verification → App passwords). "
    "A normal Gmail password will be rejected."
)


def normalize_app_password(pwd):
    return "".join((pwd or "").split())


def friendly_smtp_error(err):
    text = str(err or "")
    low = text.lower()
    if any(s in low for s in ("5.7.8", "badcredentials", "username and password not accepted", "535")):
        return AUTH_HELP
    if "5.7.9" in low or "please log in with your web browser" in low:
        return AUTH_HELP
    return text.split("\n")[0][:180] or "Email send failed"


def get_smtp_settings():
    conn = get_db()
    row = conn.execute("SELECT * FROM smtp_settings WHERE id=1").fetchone()
    conn.close()
    if not row:
        return {"from_email": "", "app_password": "", "smtp_host": "smtp.gmail.com", "smtp_port": 587}
    return {
        "from_email": (row["from_email"] or "").strip(),
        "app_password": normalize_app_password(row["app_password"] or ""),
        "smtp_host": (row["smtp_host"] or "smtp.gmail.com").strip(),
        "smtp_port": int(row["smtp_port"] or 587),
    }


def save_smtp_settings(from_email, app_password, smtp_host="smtp.gmail.com", smtp_port=587):
    conn = get_db()
    existing = conn.execute("SELECT app_password FROM smtp_settings WHERE id=1").fetchone()
    pwd = normalize_app_password(app_password)
    if not pwd and existing:
        pwd = normalize_app_password(existing["app_password"] or "")
    conn.execute(
        """
        INSERT INTO smtp_settings (id, from_email, app_password, smtp_host, smtp_port)
        VALUES (1, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            from_email=excluded.from_email,
            app_password=excluded.app_password,
            smtp_host=excluded.smtp_host,
            smtp_port=excluded.smtp_port
        """,
        ((from_email or "").strip(), pwd, smtp_host or "smtp.gmail.com", int(smtp_port or 587)),
    )
    conn.commit()
    conn.close()


def send_mail(to_email, subject, body, snapshot_path=""):
    cfg = get_smtp_settings()
    user = cfg["from_email"]
    password = cfg["app_password"]
    if not user or not password:
        return False, "Set Gmail address and app password in System → Local"
    if not to_email:
        return False, "No recipient email"

    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if snapshot_path:
        fpath = os.path.join(ALERTS_DIR, os.path.basename(snapshot_path))
        if os.path.isfile(fpath):
            try:
                with open(fpath, "rb") as f:
                    img = MIMEImage(f.read(), _subtype="jpeg")
                img.add_header("Content-Disposition", "attachment", filename=os.path.basename(fpath))
                msg.attach(img)
            except Exception as e:
                logger.debug("Could not attach snapshot: %s", e)

    host = cfg["smtp_host"] or "smtp.gmail.com"
    port = int(cfg["smtp_port"] or 587)
    payload = msg.as_string()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as smtp:
                smtp.ehlo()
                smtp.login(user, password)
                smtp.sendmail(user, [to_email], payload)
        else:
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(user, password)
                smtp.sendmail(user, [to_email], payload)
        return True, ""
    except smtplib.SMTPAuthenticationError as e:
        logger.error("SMTP auth failed: %s", e)
        return False, friendly_smtp_error(e)
    except Exception as e:
        logger.error("SMTP send failed: %s", e)
        return False, friendly_smtp_error(e)


def send_alert_email(to_email, engineer_name, alert):
    subject = f"[Vision AI] {alert.get('detection_type', 'alert')} on {alert.get('camera_id', '')}"
    body = (
        f"Hello {engineer_name or 'Engineer'},\n\n"
        f"An alert was assigned to you.\n\n"
        f"Camera: {alert.get('camera_id')}\n"
        f"Type: {alert.get('detection_type')}\n"
        f"Severity: {alert.get('severity')}\n"
        f"Time: {alert.get('created_at')}\n\n"
        f"Description:\n{alert.get('message') or '(no description)'}\n\n"
        f"— Vision AI\n"
    )
    return send_mail(to_email, subject, body, alert.get("snapshot_path") or "")


def send_test_email():
    cfg = get_smtp_settings()
    if not cfg["from_email"]:
        return False, "Enter your Gmail address first"
    body = (
        "This is a test from Vision AI.\n"
        "If you received this, SMTP is working. You can assign alerts to engineers.\n"
    )
    return send_mail(cfg["from_email"], "[Vision AI] SMTP test", body)
