"""Compatibility imports for the pipelines moved to :mod:`backend.module_1`."""

from backend.module_1.chat import ask_question, clear_chat_history
from backend.module_1.ingest import ingest_pdf

__all__ = ["ask_question", "clear_chat_history", "ingest_pdf"]
