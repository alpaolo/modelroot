"""User question validation and prompt-injection hardening for graph chat."""

import re

PROMPT_INJECTION_PATTERN = re.compile(
    r"("
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions|"
    r"disregard\s+(all\s+)?(previous|above|prior)|"
    r"you\s+are\s+now|"
    r"new\s+instructions?:|"
    r"system\s*prompt|"
    r"<\s*/?\s*system\s*>|"
    r"developer\s+message|"
    r"override\s+(the\s+)?(rules|instructions)|"
    r"forget\s+(everything|all|your)\s+(rules|instructions)|"
    r"do\s+not\s+follow\s+(the\s+)?(schema|rules)|"
    r"jailbreak"
    r")",
    re.IGNORECASE,
)
EXCESSIVE_WHITESPACE_PATTERN = re.compile(r"\s+")


class ChatRequestLimitError(ValueError):
    """Raised when a chat request exceeds configured safety limits."""


def normalize_user_question(question_text: str) -> str:
    normalized = (question_text or "").replace("\x00", " ")
    normalized = EXCESSIVE_WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def detect_prompt_injection_attempt(question_text: str) -> bool:
    return bool(PROMPT_INJECTION_PATTERN.search(question_text))


def validate_user_question(
    question_text: str,
    *,
    max_question_chars: int,
    max_question_words: int,
    block_prompt_injection: bool = True,
) -> str:
    normalized_question = normalize_user_question(question_text)
    if not normalized_question:
        raise ChatRequestLimitError("Please enter a question.")

    if len(normalized_question) > max_question_chars:
        raise ChatRequestLimitError(
            f"Question is too long (max {max_question_chars} characters)."
        )

    word_count = len(normalized_question.split())
    if word_count > max_question_words:
        raise ChatRequestLimitError(
            f"Question has too many words (max {max_question_words})."
        )

    if block_prompt_injection and detect_prompt_injection_attempt(normalized_question):
        raise ChatRequestLimitError(
            "Question contains disallowed instruction patterns."
        )

    return normalized_question
