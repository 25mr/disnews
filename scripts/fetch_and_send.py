import os
import re
import json
import time
import logging
import random
import requests
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from pathlib import Path

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%Y-%m-%d %H:%M:%S [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
BLOG_URL = "https://disroot.org/blog"
BASE_URL = "https://disroot.org"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MAILEROO_API_URL = "https://smtp.maileroo.com/api/v2/emails"
MAX_CHARS = 6000
MAX_RETRIES = 5
PAUSE_BETWEEN_SEGMENTS = 15
BEIJING_TZ = timezone(timedelta(hours=8))

# ─── Secrets from environment ───────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MAILEROO_API_KEY = os.environ.get("MAILEROO_API_KEY", "")
MAIL_FROM = os.environ.get("MAIL_FROM", "")
MAIL_TO = os.environ.get("MAIL_TO", "")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Fetch & Parse
# ═══════════════════════════════════════════════════════════════════════════

def fetch_page(url):
    """Fetch a web page and return its HTML text."""
    try:
        logger.info(f"Fetching: {url}")
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DisrootNewsletter/1.0)"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def get_latest_article_url(html):
    """Extract the URL of the latest article from the blog listing page."""
    soup = BeautifulSoup(html, "html.parser")

    # Method 1: format-bubble link
    bubble = soup.select_one(".post.format-standard .format-bubble")
    if bubble and bubble.get("href"):
        href = bubble["href"]
        return BASE_URL + href if href.startswith("/") else href

    # Method 2: post-title link
    title_a = soup.select_one(".post-title a")
    if title_a and title_a.get("href"):
        href = title_a["href"]
        return BASE_URL + href if href.startswith("/") else href

    return None


def parse_article(html, article_url):
    """Parse the article page → title / date / content_html / url."""
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_el = soup.select_one(".post-title a")
    title = title_el.get_text(strip=True) if title_el else "Unknown Title"

    # Date
    date_el = soup.select_one(".post-date")
    date_str = date_el.get_text(strip=True) if date_el else ""

    # Content
    content_el = soup.select_one(".post-content")
    if content_el is None:
        content_html = ""
    else:
        # Clean images: remove size attributes, enforce responsive
        for img in content_el.find_all("img"):
            for attr in ("width", "height"):
                img.attrs.pop(attr, None)
            img["style"] = "max-width:100%;height:auto;"
            src = img.get("src", "")
            if src.startswith("/"):
                img["src"] = BASE_URL + src

        # Fix relative links
        for a in content_el.find_all("a"):
            href = a.get("href", "")
            if href.startswith("/"):
                a["href"] = BASE_URL + href

        content_html = content_el.decode_contents()

    return {
        "title": title,
        "date": date_str,
        "content_html": content_html,
        "url": article_url,
    }


def parse_date_str(date_str):
    """'12th Apr 2026' → '2026-04-12'"""
    try:
        cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str)
        return datetime.strptime(cleaned, "%d %b %Y").strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Failed to parse date '{date_str}': {e}")
        return date_str


# ═══════════════════════════════════════════════════════════════════════════
# 2. Translation
# ═══════════════════════════════════════════════════════════════════════════

def split_html_content(html_str, max_chars=MAX_CHARS):
    """Split HTML at block-level tag boundaries, each chunk ≤ max_chars."""
    soup = BeautifulSoup(html_str, "html.parser")
    children = list(soup.children)
    if not children:
        return [html_str] if html_str else []

    chunks = []
    current = []
    current_len = 0

    for child in children:
        child_str = str(child)
        child_len = len(child_str)
        if current_len + child_len > max_chars and current:
            chunks.append("".join(str(c) for c in current))
            current = []
            current_len = 0
        current.append(child)
        current_len += child_len

    if current:
        chunks.append("".join(str(c) for c in current))

    return chunks if chunks else [html_str]


def translate_text(text, api_key):
    """Translate a single chunk via DeepSeek.  Returns translated string or None."""
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not set, skipping translation")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个专业的翻译引擎。请将以下 HTML 内容翻译为简体中文。"
                    "保留所有 HTML 标签、链接(href)、图片(src/alt)属性不变。"
                    "仅翻译标签之间的文本内容。保持原文的格式和结构。"
                    "不要添加任何解释或说明，直接输出翻译后的 HTML。"
                ),
            },
            {"role": "user", "content": text},
        ],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Translation attempt {attempt}/{MAX_RETRIES}")
            resp = requests.post(
                DEEPSEEK_API_URL, headers=headers, json=payload, timeout=300
            )

            if resp.status_code == 200:
                content = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content:
                    logger.info("Translation successful")
                    return content
                logger.warning("Empty translation response, retrying…")

            elif resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                try:
                    wait = float(ra)
                except (TypeError, ValueError):
                    wait = 2 ** attempt + random.uniform(0, 1)
                logger.warning(f"Rate limited (429), waiting {wait:.1f}s")
                time.sleep(wait)
                continue

            elif resp.status_code in (400, 401, 403):
                logger.error(
                    f"Non-retryable error {resp.status_code}: {resp.text[:500]}"
                )
                return None

            else:
                logger.warning(
                    f"Unexpected status {resp.status_code}: {resp.text[:500]}"
                )
                wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait)
                continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt}: {e}")
            wait = 2 ** attempt + random.uniform(0, 1)
            time.sleep(wait)
            continue

    logger.error("Translation failed after all retries")
    return None


def translate_content(content_html, api_key):
    """Translate article content.  Returns (html, success_bool)."""
    if not content_html:
        return "", True
    if not api_key:
        return content_html, False

    text_len = len(content_html)
    logger.info(f"Content length: {text_len} chars")

    # Short text → single call
    if text_len < MAX_CHARS:
        result = translate_text(content_html, api_key)
        return (result, True) if result is not None else (content_html, False)

    # Long text → chunked
    chunks = split_html_content(content_html, MAX_CHARS)
    logger.info(f"Split into {len(chunks)} chunks")

    all_ok = True
    translated = []
    for i, chunk in enumerate(chunks):
        result = translate_text(chunk, api_key)
        if result is not None:
            translated.append(result)
        else:
            logger.warning(f"Chunk {i+1}/{len(chunks)} failed, keeping original")
            all_ok = False
            translated.append(chunk)

        if i < len(chunks) - 1:
            logger.info(f"Pausing {PAUSE_BETWEEN_SEGMENTS}s before next chunk…")
            time.sleep(PAUSE_BETWEEN_SEGMENTS)

    return "".join(translated), all_ok


# ═══════════════════════════════════════════════════════════════════════════
# 3. Email
# ═══════════════════════════════════════════════════════════════════════════

def _email_wrap(inner_content, beijing_now):
    """Common HTML shell with header / footer."""
    now_str = beijing_now.strftime("%Y-%m-%d %H:%M UTC+8")
    return f"""<!DOCTYPE html>
<html lang="zh-CN" xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{{margin:0;padding:0;background-color:#f3f4f6;-webkit-text-size-adjust:100%}}
  img{{max-width:100%!important;height:auto!important}}
  a{{color:#2563eb}}
</style>
</head>
<body style="margin:0;padding:0;background-color:#f3f4f6;">
<table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#f3f4f6;">
<tr><td align="center" style="padding:10px 0;">
<table width="600" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;width:100%;">

<!-- HEADER -->
<tr>
<td style="background:linear-gradient(135deg,#0F172A 0%,#1E293B 100%);padding:24px 20px;text-align:center;">
  <h1 style="margin:0;color:#ffffff;font-family:Arial,sans-serif;font-size:22px;">🐾 DisNews</h1>
</td>
</tr>

<!-- BODY -->
<tr>
<td style="background-color:#ffffff;padding:24px 20px;font-family:Arial,sans-serif;">
{inner_content}
</td>
</tr>

<!-- FOOTER -->
<tr>
<td style="background:linear-gradient(135deg,#0F172A 0%,#1E293B 100%);padding:20px;text-align:center;">
  <p style="margin:0;color:#94a3b8;font-size:12px;font-family:Arial,sans-serif;">Updated at {now_str}</p>
</td>
</tr>

</table>
</td></tr></table>
</body>
</html>"""


def build_email_html(title, date_str, url, english_html, chinese_html,
                     beijing_now, translation_ok):
    date_display = parse_date_str(date_str)

    en_section = f"""
  <h2 style="margin:0 0 8px 0;font-size:18px;color:#111827;">{title}</h2>
  <p style="margin:0 0 4px 0;font-size:14px;color:#6b7280;">📅 {date_display}</p>
  <p style="margin:0 0 16px 0;font-size:13px;"><a href="{url}" style="color:#2563eb;word-break:break-all;">{url}</a></p>
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:16px 0;">
  <h3 style="margin:0 0 12px 0;font-size:16px;color:#111827;">📖 ENGLISH</h3>
  <div style="color:#111827;font-size:14px!important;line-height:1.6!important;">
    {english_html}
  </div>"""

    if translation_ok and chinese_html:
        en_section += f"""
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0;">
  <h3 style="margin:0 0 12px 0;font-size:16px;color:#374151;">🤖 中文翻译</h3>
  <div style="color:#374151;font-size:14px!important;line-height:1.6!important;">
    {chinese_html}
  </div>"""

    return _email_wrap(en_section, beijing_now)


def build_no_article_html(beijing_now):
    inner = """<p style="margin:0;font-size:16px;color:#6b7280;text-align:center;padding:20px 0;">💤 No articles today.</p>"""
    return _email_wrap(inner, beijing_now)


def build_plain_text(title, date_str, url, english_html, chinese_html,
                     translation_ok):
    en_text = BeautifulSoup(english_html or "", "html.parser").get_text("\n").strip()
    parts = [
        "📖 ENGLISH\n",
        f"Title: {title}",
        f"Date: 📅 {date_str}",
        f"URL: {url}\n",
        en_text,
    ]
    if translation_ok and chinese_html:
        zh_text = BeautifulSoup(chinese_html, "html.parser").get_text("\n").strip()
        parts.append(f"\n---\n\n🤖 中文翻译\n\n{zh_text}")
    return "\n".join(parts)


def send_email(subject, html_body, plain_body, api_key, from_addr, to_addrs):
    if not api_key or not from_addr or not to_addrs:
        logger.error("Email credentials not configured")
        return False

    to_list = [
        {"address": a.strip(), "display_name": a.strip().split("@")[0]}
        for a in to_addrs if a.strip()
    ]
    if not to_list:
        logger.error("No valid recipients")
        return False

    payload = {
        "from": {"address": from_addr, "display_name": "Newsletter"},
        "to": to_list,
        "subject": subject,
        "plain": plain_body,
        "html": html_body,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        logger.info(f"Sending email to {[t['address'] for t in to_list]}")
        resp = requests.post(MAILEROO_API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            logger.info("Email sent successfully")
            return True
        logger.error(f"Email send failed: {resp.status_code} {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Email send error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 4. GitHub Pages data
# ═══════════════════════════════════════════════════════════════════════════

def update_github_pages(article, repo_path="."):
    try:
        data_dir = Path(repo_path) / "docs"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / "data.json"

        records = []
        if data_file.exists():
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, IOError):
                records = []

        if article["url"] not in [r.get("url") for r in records]:
            records.insert(0, {
                "title": article["title"],
                "url": article["url"],
                "date": parse_date_str(article["date"]),
            })
            records = records[:10]
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated GitHub Pages data ({len(records)} records)")
        else:
            logger.info("Article already in records, skipping update")
    except Exception as e:
        logger.error(f"Failed to update GitHub Pages: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 50)
    logger.info("Disroot Blog Newsletter – Starting")
    logger.info("=" * 50)

    beijing_now = datetime.now(BEIJING_TZ)
    subject = f"🐾 DisNews - {beijing_now.strftime('%Y-%m-%d')}"
    to_addrs = [a.strip() for a in MAIL_TO.split(",") if a.strip()] if MAIL_TO else []

    def send_no_article():
        h = build_no_article_html(beijing_now)
        send_email(subject, h, "💤 No articles today.",
                   MAILEROO_API_KEY, MAIL_FROM, to_addrs)

    # ── Step 1: blog listing ─────────────────────────────────────────────
    blog_html = fetch_page(BLOG_URL)
    if not blog_html:
        logger.error("Failed to fetch blog page")
        send_no_article()
        return

    # ── Step 2: latest article URL ──────────────────────────────────────
    article_url = get_latest_article_url(blog_html)
    if not article_url:
        logger.error("Failed to find article URL")
        send_no_article()
        return
    logger.info(f"Latest article URL: {article_url}")

    # ── Step 3: article page ────────────────────────────────────────────
    article_html = fetch_page(article_url)
    if not article_html:
        logger.error("Failed to fetch article page")
        send_no_article()
        return

    # ── Step 4: parse ───────────────────────────────────────────────────
    article = parse_article(article_html, article_url)
    logger.info(f"Title: {article['title']}")
    logger.info(f"Date : {article['date']}")
    logger.info(f"Content length: {len(article['content_html'])} chars")

    # ── Step 5: translate ───────────────────────────────────────────────
    translated_html, translation_ok = translate_content(
        article["content_html"], DEEPSEEK_API_KEY
    )
    if translation_ok:
        logger.info("Translation completed successfully")
    else:
        logger.warning("Translation failed – will send English-only email")

    chinese_html = translated_html if translation_ok else ""

    # ── Step 6: build email ─────────────────────────────────────────────
    email_html = build_email_html(
        article["title"], article["date"], article["url"],
        article["content_html"], chinese_html, beijing_now, translation_ok,
    )
    plain_body = build_plain_text(
        article["title"], article["date"], article["url"],
        article["content_html"], chinese_html, translation_ok,
    )

    # ── Step 7: send ────────────────────────────────────────────────────
    send_email(subject, email_html, plain_body,
               MAILEROO_API_KEY, MAIL_FROM, to_addrs)

    # ── Step 8: GitHub Pages ────────────────────────────────────────────
    update_github_pages(article)

    logger.info("Done")


if __name__ == "__main__":
    main()
