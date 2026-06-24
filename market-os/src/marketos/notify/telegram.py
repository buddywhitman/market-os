"""Telegram push notifications — the delivery mechanism for Phase 5 briefings/alerts.
Free, no paid tier, simple bot API. Credentials (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
come from the server's .env, same pattern as every other credential in this project.

To set up (one-time, takes ~2 minutes, no cost):
  1. Message @BotFather on Telegram, send /newbot, follow the prompts -> get a bot token.
  2. Message your new bot anything (it can't message you first).
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates in a browser — your chat_id is
     in the JSON response under message.chat.id.
  4. Add both to the server's .env as TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import os

import requests

_TIMEOUT = 10
_MAX_MESSAGE_LEN = 4000  # Telegram's real limit is 4096; leave headroom for markdown escaping


def send_message(text: str, *, bot_token: str | None = None, chat_id: str | None = None) -> dict:
    """Send `text` via Telegram. Reads credentials from env if not passed explicitly.
    Never raises for a missing-credential or network failure — returns
    {"sent": False, "error": ...} instead, since a failed notification should degrade
    gracefully (log it, don't crash the job that was trying to send it), the same pattern
    as every fetcher in this codebase.
    """
    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return {"sent": False, "error": "missing_credentials"}

    # Telegram caps message length; split rather than silently truncate a long briefing.
    chunks = [text[i:i + _MAX_MESSAGE_LEN] for i in range(0, len(text), _MAX_MESSAGE_LEN)] or [""]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    for chunk in chunks:
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": chunk,
                                        "parse_mode": "Markdown"}, timeout=_TIMEOUT)
            payload = r.json()
        except (requests.RequestException, ValueError) as exc:
            return {"sent": False, "error": f"network_failure: {exc}"}
        if not payload.get("ok"):
            return {"sent": False, "error": payload.get("description", "unknown_telegram_error")}
    return {"sent": True, "chunks": len(chunks)}
