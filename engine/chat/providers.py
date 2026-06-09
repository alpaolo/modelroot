"""LLM provider factory for ModelRoot graph chat."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_CHAT_LLM_PROVIDERS = ("gemini", "deepseek", "groq", "openrouter")

_PROVIDER_SPECS = {
    "gemini": {
        "api_key_fields": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "model_field": "GEMINI_CHAT_MODEL",
        "default_model": "gemini-2.0-flash",
        "setup_hint": "https://aistudio.google.com/apikey",
    },
    "deepseek": {
        "api_key_fields": ("DEEPSEEK_API_KEY",),
        "model_field": "DEEPSEEK_CHAT_MODEL",
        "default_model": "deepseek-chat",
        "setup_hint": "https://platform.deepseek.com/api_keys",
    },
    "groq": {
        "api_key_fields": ("GROQ_API_KEY",),
        "model_field": "GROQ_CHAT_MODEL",
        "default_model": "llama-3.3-70b-versatile",
        "setup_hint": "https://console.groq.com/keys",
    },
    "openrouter": {
        "api_key_fields": ("OPENROUTER_API_KEY",),
        "model_field": "OPENROUTER_CHAT_MODEL",
        "default_model": "deepseek/deepseek-chat",
        "setup_hint": "https://openrouter.ai/keys",
    },
}


class ChatProviderConfigError(ValueError):
    """Raised when chat provider configuration is invalid."""


def normalize_chat_provider(provider_name: str) -> str:
    normalized_provider = (provider_name or "").strip().lower()
    if normalized_provider not in SUPPORTED_CHAT_LLM_PROVIDERS:
        supported = ", ".join(SUPPORTED_CHAT_LLM_PROVIDERS)
        raise ChatProviderConfigError(
            f"Unsupported CHAT_LLM_PROVIDER '{provider_name}'. Use one of: {supported}."
        )
    return normalized_provider


def resolve_active_chat_provider(config_module: Any) -> str:
    return normalize_chat_provider(
        getattr(config_module, "CHAT_LLM_PROVIDER", "deepseek")
    )


def resolve_provider_api_key(config_module: Any, provider_name: str) -> str:
    provider_spec = _PROVIDER_SPECS[provider_name]
    for api_key_field in provider_spec["api_key_fields"]:
        api_key_value = getattr(config_module, api_key_field, "") or ""
        if api_key_value.strip():
            return api_key_value.strip()
    key_fields = " or ".join(provider_spec["api_key_fields"])
    return ""


def resolve_provider_model_name(config_module: Any, provider_name: str) -> str:
    provider_spec = _PROVIDER_SPECS[provider_name]
    model_name = getattr(config_module, provider_spec["model_field"], "") or ""
    return model_name.strip() or provider_spec["default_model"]


def get_provider_setup_hint(provider_name: str) -> str:
    return _PROVIDER_SPECS[provider_name]["setup_hint"]


def get_active_provider_status(config_module: Any) -> dict[str, str]:
    provider_name = resolve_active_chat_provider(config_module)
    api_key = resolve_provider_api_key(config_module, provider_name)
    return {
        "provider": provider_name,
        "model": resolve_provider_model_name(config_module, provider_name),
        "api_key_configured": bool(api_key),
        "setup_hint": get_provider_setup_hint(provider_name),
    }


def build_chat_llm(
    config_module: Any,
    provider_name: str,
    *,
    temperature: float,
    max_output_tokens: int,
) -> BaseChatModel:
    normalized_provider = normalize_chat_provider(provider_name)
    api_key = resolve_provider_api_key(config_module, normalized_provider)
    if not api_key:
        provider_spec = _PROVIDER_SPECS[normalized_provider]
        key_fields = " or ".join(provider_spec["api_key_fields"])
        raise ChatProviderConfigError(
            f"Set {key_fields} in .env/config.py for provider '{normalized_provider}' "
            f"({provider_spec['setup_hint']})."
        )

    model_name = resolve_provider_model_name(config_module, normalized_provider)

    if normalized_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            google_api_key=api_key,
        )

    if normalized_provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        api_base = getattr(
            config_module,
            "DEEPSEEK_API_BASE",
            "https://api.deepseek.com/v1",
        )
        return ChatDeepSeek(
            model=model_name,
            temperature=temperature,
            max_tokens=max_output_tokens,
            api_key=api_key,
            api_base=api_base,
        )

    if normalized_provider == "groq":
        from langchain_openai import ChatOpenAI

        api_base = getattr(
            config_module,
            "GROQ_API_BASE",
            "https://api.groq.com/openai/v1",
        )
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_output_tokens,
            api_key=api_key,
            base_url=api_base,
        )

    from langchain_openai import ChatOpenAI

    api_base = getattr(
        config_module,
        "OPENROUTER_API_BASE",
        "https://openrouter.ai/api/v1",
    )
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
        api_key=api_key,
        base_url=api_base,
        default_headers={
            "HTTP-Referer": getattr(config_module, "OPENROUTER_HTTP_REFERER", "https://modelroot.local"),
            "X-Title": getattr(config_module, "OPENROUTER_APP_TITLE", "ModelRoot"),
        },
    )
