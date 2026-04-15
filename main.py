#!/usr/bin/env python3
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import escape as html_escape
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

BASE_URL = "https://disroot.org"
BLOG_URL = "https://disroot.org/blog"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-reasoner"
MAILEROO_API_URL = "https://smtp.maileroo.com/api/v2/emails"

DIRECT_TRANSLATE_LIMIT = 6000
CHUNK_TARGET = 5000
TRANSLATION_RETRIES = 5
CHUNK_PAUSE_SECONDS = 15

TZ_BJ = timezone(timedelta(hours=8))

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"
HISTORY_FILE = DATA_DIR / "articles.json"
PAGE_FILE = DOCS_DIR / "index.html"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 DisNewsBot/1.0"
)

UNWANTED_TAGS = [
    "script", "style", "noscript", "iframe", "object", "embed",
    "form", "button", "input", "svg", "video", "audio", "source",
    "canvas"
]

BLOCK_TAGS = {
    "p", "div", "section", "article", "blockquote", "pre",
    "ul", "ol", "li", "table", "thead", "tbody", "tr", "td",
    "th", "figure", "figcaption", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
}


class BeijingFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=TZ_BJ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


logger = logging.getLogger("disnews")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(BeijingFormatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.handlers.clear()
logger.addHandler(_handler)


class NonRetryableTranslationError(Exception):
    pass


def bj_now() -> datetime:
    return datetime.now(TZ_BJ)


def bj_date_str() -> str:
    return bj_now().strftime("%Y-%m-%d")


def bj_datetime_str() -> str:
    return bj_now().strftime("%Y-%m-%d %H:%M")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if not HISTORY_FILE.exists():
        return {"updated_at": "", "records": []}

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {"updated_at": "", "records": data}
        if isinstance(data, dict):
            return {
                "updated_at": data.get("updated_at", ""),
                "records": data.get("records", []),
            }
    except Exception as exc:
        logger.warning("Failed to load history file, start fresh: %s", exc)

    return {"updated_at": "", "records": []}


def save_state(state: dict) -> None:
    HISTORY_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_page(html: str) -> None:
    PAGE_FILE.write_text(html, encoding="utf-8")


def append_style(existing: str | None, new_style: str) -> str:
    existing = (existing or "").strip()
    new_style = (new_style or "").strip()
    if existing and not existing.endswith(";"):
        existing += ";"
    if new_style and not new_style.endswith(";"):
        new_style += ";"
    if existing:
        return existing + new_style
    return new_style


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        wait = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, wait)
    except Exception:
        return None


def backoff_seconds(attempt: int) -> float:
    # 指数退避 + 抖动
    base = 2 ** (attempt - 1)
    jitter = random.uniform(0.5, 1.5)
    return min(60.0, base + jitter)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def safe_get(session: requests.Session, url: str, timeout: int = 30) -> requests.Response:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def get_latest_article_url(session: requests.Session) -> str:
    logger.info("Fetching homepage: %s", BLOG_URL)
    resp = safe_get(session, BLOG_URL, timeout=30)
    soup = BeautifulSoup(resp.text, "lxml")

    first_post = soup.select_one("div.posts div.post") or soup.select_one("div.post")
    if not first_post:
        raise ValueError("No post block found on homepage")

    link = (
        first_post.select_one("h2.post-title a[href]")
        or first_post.select_one(".post-bubbles a[href]")
        or first_post.select_one("a[href^='/blog/']")
    )
    if not link:
        raise ValueError("No article link found on homepage")

    href = link.get("href", "").strip()
    if not href:
        raise ValueError("Empty article href")

    return urljoin(BASE_URL, href)


def sanitize_img_tag(img: Tag, base_url: str = BASE_URL) -> None:
    src = img.get("src") or img.get("data-src") or img.get("data-original")
    if src:
        img["src"] = urljoin(base_url, src.strip())

    # 移除所有可能导致撑破屏幕的属性
    for attr in [
        "srcset", "sizes", "width", "height",
        "loading", "decoding", "data-src", "data-original",
        "style"
    ]:
        img.attrs.pop(attr, None)

    # 强制响应式显示
    img["style"] = (
        "display:block !important;"
        "width:100% !important;"
        "max-width:100% !important;"
        "height:auto !important;"
        "margin:12px auto !important;"
        "border:0 !important;"
        "outline:none !important;"
        "box-sizing:border-box !important;"
    )
    img["border"] = "0"
    img["align"] = "center"


def clean_article_html(fragment_html: str, base_url: str) -> str:
    soup = BeautifulSoup(fragment_html, "lxml")

    for tag in soup.find_all(UNWANTED_TAGS):
        tag.decompose()

    for a in soup.find_all("a"):
        href = a.get("href")
        if href:
            a["href"] = urljoin(base_url, href.strip())

    for img in soup.find_all("img"):
        sanitize_img_tag(img, base_url)

    # 也清理 picture/source 的 srcset，避免某些客户端选中超大图
    for source in soup.find_all("source"):
        for attr in ["srcset", "sizes", "width", "height"]:
            source.attrs.pop(attr, None)

    return "".join(str(x) for x in soup.contents).strip()


def fetch_article(session: requests.Session, article_url: str) -> dict:
    logger.info("Fetching article: %s", article_url)
    resp = safe_get(session, article_url, timeout=30)
    soup = BeautifulSoup(resp.text, "lxml")

    title_node = soup.select_one("h2.post-title") or soup.select_one("h1.post-title") or soup.select_one("h1")
    date_node = soup.select_one(".post-meta .post-date") or soup.select_one(".post-date")
    content_node = soup.select_one(".post-content")

    if not title_node:
        raise ValueError("Article title not found")
    if not content_node:
        raise ValueError("Article content not found")

    title = title_node.get_text(" ", strip=True)
    published_date = date_node.get_text(" ", strip=True) if date_node else ""
    content_html = clean_article_html(content_node.decode_contents(), BASE_URL)

    return {
        "url": article_url,
        "title": title,
        "published_date": published_date,
        "content_html": content_html,
    }


def html_text_length(html_fragment: str) -> int:
    soup = BeautifulSoup(html_fragment, "lxml")
    return len(soup.get_text(" ", strip=True))


def split_plain_text(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # 优先按中英文句末切分
    sentences = re.split(r"(?<=[。！？.!?])\s+", text)
    if len(sentences) == 1:
        # 再按单词
        import textwrap
        sentences = textwrap.wrap(
            text,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )

    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        candidate = f"{current} {s}".strip() if current else s
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(s) <= max_chars:
                current = s
            else:
                # 再强制切块
                for i in range(0, len(s), max_chars):
                    part = s[i:i + max_chars].strip()
                    if part:
                        chunks.append(part)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def attrs_to_str(attrs: dict) -> str:
    if not attrs:
        return ""
    parts = []
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v)
        parts.append(f'{k}="{html_escape(str(v), quote=True)}"')
    return (" " + " ".join(parts)) if parts else ""


def wrap_tag(tag_name: str, attrs: dict, inner_html: str) -> str:
    return f"<{tag_name}{attrs_to_str(attrs)}>{inner_html}</{tag_name}>"


def split_node_to_units(node, max_chars: int) -> list[str]:
    if isinstance(node, NavigableString):
        text = str(node)
        return [text] if text.strip() else []

    if not isinstance(node, Tag):
        return []

    if node.name in UNWANTED_TAGS:
        return []

    html = str(node)
    if html_text_length(html) <= max_chars:
        return [html]

    # 对 ul/ol 做更稳妥的分组
    if node.name in {"ul", "ol"}:
        items = [str(li) for li in node.find_all("li", recursive=False)]
        if not items:
            return [html]
        groups = []
        current = []
        current_len = 0
        for item in items:
            item_len = html_text_length(item)
            if current and current_len + item_len > max_chars:
                groups.append(wrap_tag(node.name, node.attrs, "".join(current)))
                current = [item]
                current_len = item_len
            else:
                current.append(item)
                current_len += item_len
        if current:
            groups.append(wrap_tag(node.name, node.attrs, "".join(current)))
        return groups

    # 其他块级容器：尽量按子节点拆分
    child_tags = [c for c in node.children if isinstance(c, Tag)]
    if child_tags:
        units = []
        for child in node.children:
            units.extend(split_node_to_units(child, max_chars))
        return units if units else [html]

    # 纯文本块：拆成多个同标签块
    text = node.get_text(" ", strip=True)
    if not text:
        return [html]
    if len(text) <= max_chars:
        return [html]
    segments = split_plain_text(text, max_chars)
    if len(segments) <= 1:
        return [html]
    return [wrap_tag(node.name, node.attrs, html_escape(seg)) for seg in segments]


def split_html_for_translation(html_fragment: str, max_chars: int) -> list[str]:
    soup = BeautifulSoup(html_fragment, "lxml")
    root = soup.body if soup.body else soup
    units = []

    for child in root.contents:
        units.extend(split_node_to_units(child, max_chars))

    chunks = []
    current = []
    current_len = 0

    for unit in units:
        unit_len = html_text_length(unit)
        if unit_len > max_chars:
            if current:
                chunks.append("".join(current))
                current = []
                current_len = 0
            chunks.append(unit)
            continue

        if current and current_len + unit_len > max_chars:
            chunks.append("".join(current))
            current = [unit]
            current_len = unit_len
        else:
            current.append(unit)
            current_len += unit_len

    if current:
        chunks.append("".join(current))

    return [c for c in chunks if c.strip()]


def normalize_translated_fragment(text: str) -> str:
    text = strip_code_fences(text)
    soup = BeautifulSoup(text, "lxml")
    if soup.body:
        text = "".join(str(x) for x in soup.body.contents)
    elif soup.html and soup.html.body:
        text = "".join(str(x) for x in soup.html.body.contents)
    else:
        text = "".join(str(x) for x in soup.contents) if soup.contents else text
    return text.strip()


def normalize_plain_translation(text: str) -> str:
    text = strip_code_fences(text)
    soup = BeautifulSoup(text, "lxml")
    cleaned = soup.get_text(" ", strip=True)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n\"“”'`")
    return cleaned


def build_deepseek_headers() -> dict:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def call_deepseek(messages: list[dict], max_tokens: int = 4096) -> str:
    headers = build_deepseek_headers()
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 1.3,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_error = None
    for attempt in range(1, TRANSLATION_RETRIES + 1):
        try:
            resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= TRANSLATION_RETRIES:
                raise
            wait = backoff_seconds(attempt)
            logger.warning("DeepSeek request error (attempt %s/%s), retry in %.1fs: %s", attempt, TRANSLATION_RETRIES, wait, exc)
            time.sleep(wait)
            continue

        if resp.status_code in (400, 401, 403):
            raise NonRetryableTranslationError(f"DeepSeek non-retryable error {resp.status_code}: {resp.text[:500]}")

        if resp.status_code == 429:
            retry_after = parse_retry_after(resp.headers.get("Retry-After"))
            wait = retry_after if retry_after is not None else backoff_seconds(attempt)
            logger.warning("DeepSeek rate limited (attempt %s/%s), retry in %.1fs", attempt, TRANSLATION_RETRIES, wait)
            time.sleep(wait)
            continue

        if resp.status_code >= 500 or resp.status_code in (408, 409):
            last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            if attempt >= TRANSLATION_RETRIES:
                raise last_error
            wait = backoff_seconds(attempt)
            logger.warning("DeepSeek retryable HTTP error (attempt %s/%s), retry in %.1fs: %s", attempt, TRANSLATION_RETRIES, wait, resp.status_code)
            time.sleep(wait)
            continue

        try:
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise RuntimeError("Empty DeepSeek response content")
            return content.strip()
        except Exception as exc:
            last_error = exc
            if attempt >= TRANSLATION_RETRIES:
                raise
            wait = backoff_seconds(attempt)
            logger.warning("DeepSeek response parse error (attempt %s/%s), retry in %.1fs: %s", attempt, TRANSLATION_RETRIES, wait, exc)
            time.sleep(wait)

    raise RuntimeError(f"DeepSeek translation failed: {last_error}")


def translate_text_to_zh(text: str) -> str:
    prompt = (
        "请将下面这段内容翻译成简体中文。\n"
        "要求：\n"
        "1. 只输出译文，不要解释。\n"
        "2. 保留专有名词、数字、日期、URL。\n"
        "3. 不要添加 Markdown。\n\n"
        f"{text}"
    )
    result = call_deepseek(
        [
            {"role": "system", "content": "你是专业翻译器，只输出准确的简体中文译文。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
    )
    cleaned = normalize_plain_translation(result)
    return cleaned or text


def translate_html_to_zh(html_fragment: str) -> str:
    text_len = len(BeautifulSoup(html_fragment, "lxml").get_text(" ", strip=True))

    html_prompt_prefix = (
        "请将下面的 HTML 片段翻译成简体中文。\n"
        "要求：\n"
        "1. 保留所有 HTML 标签、属性、链接和图片。\n"
        "2. 不要翻译 URL。\n"
        "3. 不要删除或新增标签，不要修改标签结构。\n"
        "4. 不要输出解释、注释、Markdown 或代码块。\n"
        "5. 输出必须是可直接嵌入邮件的 HTML 片段。\n\n"
    )

    def translate_one(fragment: str) -> str:
        result = call_deepseek(
            [
                {"role": "system", "content": "你是专业 HTML 翻译器，只输出翻译后的 HTML 片段。"},
                {"role": "user", "content": html_prompt_prefix + fragment},
            ],
            max_tokens=4096,
        )
        return normalize_translated_fragment(result)

    if text_len < DIRECT_TRANSLATE_LIMIT:
        return translate_one(html_fragment)

    chunks = split_html_for_translation(html_fragment, CHUNK_TARGET)
    if not chunks:
        return ""

    translated_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Translating chunk %s/%s (approx %s chars)", idx, len(chunks), html_text_length(chunk))
        translated_chunks.append(translate_one(chunk))
        if idx != len(chunks):
            time.sleep(CHUNK_PAUSE_SECONDS)

    return "\n".join(translated_chunks).strip()


def style_email_fragment(fragment_html: str, text_color: str) -> str:
    soup = BeautifulSoup(fragment_html, "lxml")
    root = soup.body if soup.body else soup

    # 二次清洗图片，防止翻译后又带回尺寸信息
    for img in root.find_all("img"):
        sanitize_img_tag(img, BASE_URL)

    for a in root.find_all("a"):
        a["style"] = append_style(
            a.get("style"),
            "color:#2563EB;text-decoration:underline;word-break:break-word;",
        )

    # 给可能包含图片的容器加上防溢出
    for tag in root.find_all(["figure", "figcaption", "div"]):
        tag["style"] = append_style(
            tag.get("style"),
            "max-width:100% !important;overflow:hidden;",
        )

    for tag_name in ["p", "ul", "ol", "blockquote", "pre", "table", "figure", "figcaption"]:
        for tag in root.find_all(tag_name):
            if tag_name in {"ul", "ol"}:
                extra = "margin:0 0 12px 20px;padding:0;"
            elif tag_name == "blockquote":
                extra = "margin:0 0 12px;padding:12px 16px;border-left:4px solid #CBD5E1;background:#F8FAFC;"
            elif tag_name == "pre":
                extra = "margin:0 0 12px;padding:12px;background:#F3F4F6;border-radius:8px;white-space:pre-wrap;word-break:break-word;overflow-x:auto;"
            elif tag_name == "table":
                extra = "width:100%;border-collapse:collapse;margin:0 0 12px;"
            elif tag_name in {"figure", "figcaption"}:
                extra = "margin:0 0 12px;max-width:100%;overflow:hidden;"
            else:
                extra = "margin:0 0 12px;"

            tag["style"] = append_style(
                tag.get("style"),
                f"font-size:14px !important;line-height:1.6 !important;color:{text_color};{extra}",
            )

    for tag in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        tag["style"] = append_style(
            tag.get("style"),
            f"margin:0 0 12px;line-height:1.3 !important;color:{text_color};",
        )

    for tag in root.find_all(["td", "th"]):
        tag["style"] = append_style(
            tag.get("style"),
            f"font-size:14px !important;line-height:1.6 !important;color:{text_color};padding:8px;border:1px solid #E5E7EB;",
        )

    return "".join(str(x) for x in root.contents).strip()


def text_from_html(fragment_html: str) -> str:
    soup = BeautifulSoup(fragment_html, "lxml")
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_section_html(label: str, title: str, url: str, published_date: str, body_html: str, color: str) -> str:
    safe_title = html_escape(title)
    safe_url = html_escape(url)
    safe_date = html_escape(published_date or "")
    styled_body = style_email_fragment(body_html, color)

    return f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;margin:0 0 20px;">
      <tr>
        <td style="padding:0 0 8px;">
          <div style="font-size:16px !important;line-height:1.4 !important;font-weight:700;color:{color};">{label}</div>
        </td>
      </tr>
      <tr>
        <td style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;padding:18px;overflow:hidden;">
          <div style="font-size:14px !important;line-height:1.6 !important;color:{color};word-break:break-word;overflow-wrap:anywhere;max-width:100%;overflow:hidden;">
            <p style="margin:0 0 10px;"><strong>{safe_title}</strong></p>
            <p style="margin:0 0 10px;">🔗 <a href="{safe_url}" style="color:#2563EB;text-decoration:underline;word-break:break-word;">{safe_url}</a></p>
            <p style="margin:0 0 14px;">📅 {safe_date}</p>
            <div style="font-size:14px !important;line-height:1.6 !important;color:{color};word-break:break-word;overflow-wrap:anywhere;max-width:100%;overflow:hidden;">
              {styled_body}
            </div>
          </div>
        </td>
      </tr>
    </table>
    """.strip()


def build_no_article_section() -> str:
    return """
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;margin:0 0 20px;">
      <tr>
        <td style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:16px;padding:18px;">
          <div style="font-size:14px !important;line-height:1.6 !important;color:#111827;">
            💤 No articles today.
          </div>
        </td>
      </tr>
    </table>
    """.strip()


def build_email_plain(subject_date: str, article: dict | None, translated_title: str | None, translated_body_html: str | None) -> str:
    if not article:
        return f"""🐾 DisNews - {subject_date}

💤 No articles today.

Updated at {bj_now().strftime("%Y-%m-%d %H:%M")} UTC+8
"""

    english_text = text_from_html(article["content_html"])
    lines = [
        f"🐾 DisNews - {subject_date}",
        "",
        "📖 ENGLISH",
        f"Title: {article['title']}",
        f"URL: {article['url']}",
        f"📅 {article['published_date']}",
        "",
        english_text,
    ]

    if translated_body_html:
        zh_text = text_from_html(translated_body_html)
        lines.extend([
            "",
            "🤖 中文翻译",
            f"标题：{translated_title or article['title']}",
            f"链接：{article['url']}",
            f"📅 {article['published_date']}",
            "",
            zh_text,
        ])

    lines.extend([
        "",
        f"Updated at {bj_now().strftime('%Y-%m-%d %H:%M')} UTC+8",
    ])
    return "\n".join(lines).strip() + "\n"


def build_email_html(subject_date: str, article: dict | None, translated_title: str | None, translated_body_html: str | None) -> str:
    updated_at = bj_now().strftime("%Y-%m-%d %H:%M")
    if not article:
        content_block = build_no_article_section()
    else:
        english_section = build_section_html(
            "📖 ENGLISH",
            article["title"],
            article["url"],
            article["published_date"],
            article["content_html"],
            "#111827",
        )
        if translated_body_html:
            chinese_section = build_section_html(
                "🤖 中文翻译",
                translated_title or article["title"],
                article["url"],
                article["published_date"],
                translated_body_html,
                "#374151",
            )
            content_block = english_section + chinese_section
        else:
            content_block = english_section

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🐾 DisNews - {subject_date}</title>
</head>
<body style="Margin:0;padding:0;background-color:#F3F4F6;-webkit-text-size-adjust:100%;-ms-text-size-adjust:100%;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;background-color:#F3F4F6;width:100%;">
    <tr>
      <td align="center" style="padding:0;margin:0;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;max-width:680px;width:100%;margin:0 auto;background-color:#FFFFFF;">
          <tr>
            <td style="background:linear-gradient(135deg,#0F172A 0%,#111827 100%);background-color:#0F172A;padding:24px 20px;text-align:left;">
              <div style="font-size:30px !important;line-height:1.3 !important;font-weight:700;color:#FFFFFF;">🐈 DisNews</div>
            </td>
          </tr>
          <tr>
            <td style="padding:20px;">
              {content_block}
            </td>
          </tr>
          <tr>
            <td style="background:linear-gradient(135deg,#0F172A 0%,#111827 100%);background-color:#0F172A;padding:16px 20px;text-align:center;">
              <div style="font-size:12px !important;line-height:1.6 !important;color:#E2E8F0;">Updated at {html_escape(updated_at)} UTC+8</div>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
    

def parse_recipients(mail_to: str) -> list[dict]:
    parts = [p.strip() for p in re.split(r"[,\n;]+", mail_to or "") if p.strip()]
    recipients = []

    for part in parts:
        name = ""
        addr = part

        m = re.match(r"^(.*?)\s*<([^>]+)>$", part)
        if m:
            name = m.group(1).strip().strip('"').strip("'")
            addr = m.group(2).strip()

        if "@" not in addr:
            logger.warning("Skip invalid recipient: %s", part)
            continue

        item = {"address": addr}
        if name:
            item["display_name"] = name
        recipients.append(item)

    return recipients


def send_email(subject: str, html_body: str, plain_body: str) -> None:
    maileroo_key = os.getenv("MAILEROO_API_KEY", "").strip()
    mail_from = os.getenv("MAIL_FROM", "").strip()
    mail_to = os.getenv("MAIL_TO", "").strip()

    if not maileroo_key:
        raise RuntimeError("MAILEROO_API_KEY is missing")
    if not mail_from:
        raise RuntimeError("MAIL_FROM is missing")
    if not mail_to:
        raise RuntimeError("MAIL_TO is missing")

    recipients = parse_recipients(mail_to)
    if not recipients:
        raise RuntimeError("MAIL_TO contains no valid recipients")

    payload = {
        "from": {
            "address": mail_from,
            "display_name": "Newsletter",
        },
        "to": recipients,
        "subject": subject,
        "plain": plain_body,
        "html": html_body,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {maileroo_key}",
    }

    logger.info("Sending email to %s recipient(s)", len(recipients))
    resp = requests.post(MAILEROO_API_URL, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Maileroo API failed: HTTP {resp.status_code} {resp.text[:1000]}")

    logger.info("Email sent successfully")


def update_history_with_article(state: dict, article: dict) -> dict:
    records = state.get("records", [])
    if not isinstance(records, list):
        records = []

    # 避免重复记录：按 url 去重
    records = [r for r in records if r.get("url") != article["url"]]

    new_record = {
        "title": article["title"],
        "url": article["url"],
        "published_date": article["published_date"],
        "fetched_at": bj_now().strftime("%Y-%m-%d %H:%M"),
    }
    records.insert(0, new_record)
    records = records[:10]

    state["updated_at"] = bj_now().strftime("%Y-%m-%d %H:%M")
    state["records"] = records
    return state


def render_archive_page(state: dict) -> str:
    updated_at = state.get("updated_at") or bj_now().strftime("%Y-%m-%d %H:%M")
    records = state.get("records", []) or []

    rows = []
    for idx, rec in enumerate(records[:10], start=1):
        title = html_escape(rec.get("title", ""))
        url = html_escape(rec.get("url", ""))
        published = html_escape(rec.get("published_date", ""))
        rows.append(f"""
          <tr>
            <td style="padding:12px 10px;border-bottom:1px solid #E5E7EB;color:#6B7280;width:44px;text-align:center;">{idx}</td>
            <td style="padding:12px 10px;border-bottom:1px solid #E5E7EB;word-break:break-word;">
              <a href="{url}" style="color:#1D4ED8;text-decoration:none;font-weight:600;">{title}</a>
            </td>
            <td style="padding:12px 10px;border-bottom:1px solid #E5E7EB;word-break:break-word;">
              <a href="{url}" style="color:#2563EB;text-decoration:underline;">{url}</a>
            </td>
            <td style="padding:12px 10px;border-bottom:1px solid #E5E7EB;color:#374151;white-space:nowrap;">{published}</td>
          </tr>
        """)

    if not rows:
        rows_html = """
          <tr>
            <td colspan="4" style="padding:18px;color:#6B7280;text-align:center;">No records yet.</td>
          </tr>
        """
    else:
        rows_html = "\n".join(rows)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DisNews Archive</title>
  <style>
    body {{
      margin: 0;
      background: #F3F4F6;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
                   Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      color: #111827;
    }}
    .wrap {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #E5E7EB;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(0,0,0,.04);
    }}
    .header {{
      background: linear-gradient(135deg, #0F172A 0%, #111827 100%);
      color: #fff;
      padding: 24px 20px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.3;
    }}
    .header p {{
      margin: 8px 0 0;
      color: #E2E8F0;
      font-size: 14px;
      line-height: 1.6;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    thead th {{
      background: #F9FAFB;
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid #E5E7EB;
      font-size: 13px;
      color: #374151;
    }}
    @media (max-width: 720px) {{
      .hide-mobile {{
        display: none;
      }}
      table {{
        table-layout: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <h1>🐾 DisNews Archive</h1>
        <p>Updated at {html_escape(updated_at)} UTC+8</p>
      </div>
      <div style="padding:16px;">
        <table>
          <thead>
            <tr>
              <th style="width:44px;text-align:center;">#</th>
              <th>Title</th>
              <th>URL</th>
              <th style="width:180px;" class="hide-mobile">Published date</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</body>
</html>"""


def main() -> int:
    ensure_dirs()
    session = make_session()

    subject_date = bj_date_str()
    subject = f"🐾 DisNews - {subject_date}"
    now_bj = bj_now().strftime("%Y-%m-%d %H:%M")

    state = load_state()
    article = None
    translated_title = None
    translated_body_html = None

    try:
        latest_url = get_latest_article_url(session)
        article = fetch_article(session, latest_url)
        logger.info("Latest article: %s", article["title"])
    except Exception as exc:
        logger.exception("Failed to fetch latest article: %s", exc)
        article = None

    if article:
        try:
            # 标题翻译失败不影响正文翻译；标题失败则回退英文
            try:
                translated_title = translate_text_to_zh(article["title"])
                logger.info("Title translated successfully")
            except Exception as exc:
                logger.warning("Title translation failed, fallback to original title: %s", exc)
                translated_title = article["title"]

            translated_body_html = translate_html_to_zh(article["content_html"])
            logger.info("Article body translated successfully")
        except NonRetryableTranslationError as exc:
            logger.error("Translation stopped by non-retryable error, sending English only: %s", exc)
            translated_body_html = None
            translated_title = None
        except Exception as exc:
            logger.error("Translation failed, sending English only: %s", exc)
            translated_body_html = None
            translated_title = None

        # 更新历史记录
        state = update_history_with_article(state, article)
        save_state(state)
        write_page(render_archive_page(state))
    else:
        # 没拿到文章，也要把页面更新时间刷新
        state["updated_at"] = now_bj
        save_state(state)
        write_page(render_archive_page(state))

    html_body = build_email_html(subject_date, article, translated_title, translated_body_html)
    plain_body = build_email_plain(subject_date, article, translated_title, translated_body_html)

    try:
        send_email(subject, html_body, plain_body)
    except Exception as exc:
        logger.exception("Email sending failed: %s", exc)
        return 1

    logger.info("Done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        raise SystemExit(130)
