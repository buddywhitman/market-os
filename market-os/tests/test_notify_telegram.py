"""Unit tests for notify/telegram.py — focused on graceful degradation (missing
credentials, network failure) since this is push-notification code that must never
crash the job that's trying to send a message.
"""
from __future__ import annotations

from unittest.mock import patch

from marketos.notify.telegram import send_message


def test_missing_credentials_returns_error_without_raising(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    result = send_message("hello")
    assert result["sent"] is False
    assert result["error"] == "missing_credentials"


def test_network_failure_returns_error_without_raising():
    import requests
    with patch("marketos.notify.telegram.requests.post",
              side_effect=requests.exceptions.ConnectionError("boom")):
        result = send_message("hello", bot_token="fake", chat_id="fake")
    assert result["sent"] is False
    assert "network_failure" in result["error"]


def test_successful_send_returns_sent_true():
    with patch("marketos.notify.telegram.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"ok": True}
        result = send_message("hello", bot_token="fake", chat_id="fake")
    assert result["sent"] is True


def test_telegram_api_error_returns_error_without_raising():
    with patch("marketos.notify.telegram.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"ok": False, "description": "chat not found"}
        result = send_message("hello", bot_token="fake", chat_id="fake")
    assert result["sent"] is False
    assert "chat not found" in result["error"]


def test_long_message_split_into_multiple_chunks():
    long_text = "x" * 9000  # over the 4000-char chunk size, under Telegram's hard limit x2
    with patch("marketos.notify.telegram.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"ok": True}
        result = send_message(long_text, bot_token="fake", chat_id="fake")
    assert result["sent"] is True
    assert result["chunks"] > 1
    assert mock_post.call_count == result["chunks"]
