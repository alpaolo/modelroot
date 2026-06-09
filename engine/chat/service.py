"""ModelRoot chat service: LLM Cypher generation + guarded Neo4j execution."""

from typing import Any, Callable, Optional

from engine.chat.guards import CypherGuardError, validate_read_only_cypher
from engine.chat.providers import (
    build_chat_llm,
    resolve_active_chat_provider,
)
from engine.chat.request_limits import ChatRequestLimitError, validate_user_question
from engine.chat.result_router import classify_chat_rows, rows_to_summary_json
from engine.chat.schema_prompt import build_cypher_prompt, build_summary_prompt
from engine.chat.types import ChatOutputKind, ChatResult


class ChatService:
    def __init__(
        self,
        provider_name: str,
        cypher_llm,
        summary_llm,
        max_rows: int,
        default_language: str,
        neo4j_execute: Callable[[str, Optional[dict[str, Any]]], list[dict[str, Any]]],
        max_question_chars: int,
        max_question_words: int,
        summary_max_rows: int,
        summary_max_json_chars: int,
        block_prompt_injection: bool,
    ):
        self.provider_name = provider_name
        self.max_rows = max_rows
        self.default_language = default_language
        self.neo4j_execute = neo4j_execute
        self.cypher_llm = cypher_llm
        self.summary_llm = summary_llm
        self.max_question_chars = max_question_chars
        self.max_question_words = max_question_words
        self.summary_max_rows = summary_max_rows
        self.summary_max_json_chars = summary_max_json_chars
        self.block_prompt_injection = block_prompt_injection

    @classmethod
    def from_config(
        cls,
        config_module: Any,
        neo4j_execute: Callable[[str, Optional[dict[str, Any]]], list[dict[str, Any]]],
    ) -> "ChatService":
        provider_name = resolve_active_chat_provider(config_module)
        max_rows = int(getattr(config_module, "CHAT_MAX_ROWS", 50))
        default_language = getattr(config_module, "CHAT_DEFAULT_LANGUAGE", "en")
        cypher_max_tokens = int(getattr(config_module, "CHAT_CYPHER_MAX_TOKENS", 512))
        summary_max_tokens = int(getattr(config_module, "CHAT_SUMMARY_MAX_TOKENS", 512))

        return cls(
            provider_name=provider_name,
            cypher_llm=build_chat_llm(
                config_module,
                provider_name,
                temperature=0,
                max_output_tokens=cypher_max_tokens,
            ),
            summary_llm=build_chat_llm(
                config_module,
                provider_name,
                temperature=0.2,
                max_output_tokens=summary_max_tokens,
            ),
            max_rows=max_rows,
            default_language=default_language,
            neo4j_execute=neo4j_execute,
            max_question_chars=int(getattr(config_module, "CHAT_MAX_QUESTION_CHARS", 400)),
            max_question_words=int(getattr(config_module, "CHAT_MAX_QUESTION_WORDS", 60)),
            summary_max_rows=int(getattr(config_module, "CHAT_SUMMARY_MAX_ROWS", 15)),
            summary_max_json_chars=int(
                getattr(config_module, "CHAT_SUMMARY_MAX_JSON_CHARS", 8000)
            ),
            block_prompt_injection=bool(
                getattr(config_module, "CHAT_BLOCK_PROMPT_INJECTION", True)
            ),
        )

    def _validate_question(self, question: str) -> str:
        return validate_user_question(
            question,
            max_question_chars=self.max_question_chars,
            max_question_words=self.max_question_words,
            block_prompt_injection=self.block_prompt_injection,
        )

    def _generate_cypher(self, question: str) -> str:
        prompt_text = build_cypher_prompt(question, self.max_rows)
        llm_response = self.cypher_llm.invoke(prompt_text)
        return (llm_response.content or "").strip()

    def _summarize(
        self,
        question: str,
        cypher: str,
        rows: list[dict[str, Any]],
        language: str,
    ) -> str:
        summary_prompt = build_summary_prompt(
            question=question,
            cypher=cypher,
            rows_json=rows_to_summary_json(
                rows,
                max_rows=self.summary_max_rows,
                max_json_chars=self.summary_max_json_chars,
            ),
            language=language,
        )
        llm_response = self.summary_llm.invoke(summary_prompt)
        return (llm_response.content or "").strip() or "No summary available."

    def ask(self, question: str, language: Optional[str] = None) -> ChatResult:
        raw_cypher = ""

        try:
            question_text = self._validate_question(question)
        except ChatRequestLimitError as limit_error:
            return ChatResult(
                answer_text=str(limit_error),
                output_kind=ChatOutputKind.TEXT,
                error=str(limit_error),
            )

        response_language = language or self.default_language

        try:
            raw_cypher = self._generate_cypher(question_text)
            validated_cypher = validate_read_only_cypher(raw_cypher, self.max_rows)
            rows = self.neo4j_execute(validated_cypher)
            summary_text = self._summarize(
                question_text,
                validated_cypher,
                rows,
                response_language,
            )
            return classify_chat_rows(rows, summary_text, validated_cypher)
        except CypherGuardError as guard_error:
            return ChatResult(
                answer_text=f"Query blocked for safety: {guard_error}",
                output_kind=ChatOutputKind.TEXT,
                cypher=raw_cypher,
                error=str(guard_error),
            )
        except Exception as service_error:
            return ChatResult(
                answer_text=f"Chat query failed: {service_error}",
                output_kind=ChatOutputKind.TEXT,
                cypher=raw_cypher,
                error=str(service_error),
            )
