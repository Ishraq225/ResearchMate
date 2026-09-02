"""
generator.py
============

Builds the final RAG prompt and calls an LLM to generate an answer.

WHY THE LLM PROVIDER IS ABSTRACTED
--------------------------------------
The project spec requires the app to work with either a local model
(Ollama) or an API-based model (OpenAI), without hard-coding the
whole application to one provider. We achieve this with a small
strategy pattern: `BaseLLMProvider` defines one method, `complete()`,
and each concrete provider (OllamaProvider, OpenAIProvider) implements
it differently. Everything else in the app (Generator, app.py) only
ever calls `provider.complete(...)` — it doesn't know or care which
backend is actually running.

WHAT WOULD HAPPEN WITHOUT THIS ABSTRACTION
-----------------------------------------------
Every place that calls the LLM would need an if/else on provider type,
duplicated across the codebase. Adding a third provider later (e.g.
Anthropic's API) would mean hunting down every call site instead of
writing one new class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from app.prompts import SYSTEM_PROMPT, build_rag_prompt
from app.retriever import RetrievedChunk


class GenerationError(Exception):
    """Raised when the LLM provider fails to produce a response."""


# ----------------------------------------------------------------------
# Provider abstraction
# ----------------------------------------------------------------------
class BaseLLMProvider(ABC):
    """Common interface every LLM backend must implement."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text completion for the given prompts."""
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    """Local LLM via Ollama (no API key required, runs on your machine)."""

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise GenerationError(
                "The 'ollama' package is not installed. Run: pip install ollama"
            ) from exc

        try:
            client = ollama.Client(host=self.base_url)
            response = client.chat(
    model=self.model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    options={
        "temperature": 0.2,
        "num_predict": 512,
    },
)
            return response["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                f"Ollama request failed. Is the Ollama server running at "
                f"{self.base_url} and is model '{self.model}' pulled? "
                f"(Try: `ollama pull {self.model}`). Original error: {exc}"
            ) from exc


class OpenAIProvider(BaseLLMProvider):
    """API-based LLM via OpenAI (requires OPENAI_API_KEY in .env)."""

    def __init__(self, model: str = OPENAI_MODEL, api_key: str | None = OPENAI_API_KEY):
        if not api_key:
            raise GenerationError(
                "OPENAI_API_KEY is not set. Add it to your .env file, or set "
                "LLM_PROVIDER=ollama to use a local model instead."
            )
        self.model = model
        self.api_key = api_key

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenerationError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"OpenAI request failed: {exc}") from exc


def get_llm_provider(provider_name: str = LLM_PROVIDER) -> BaseLLMProvider:
    """
    Factory: returns the configured LLM provider instance.

    This is the ONLY place in the codebase that decides which provider
    class to instantiate — everything else depends only on
    BaseLLMProvider's interface.
    """
    provider_name = (provider_name or "").lower()
    if provider_name == "ollama":
        return OllamaProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    raise GenerationError(
        f"Unknown LLM_PROVIDER '{provider_name}'. Use 'ollama' or 'openai'."
    )


# ----------------------------------------------------------------------
# Generation orchestration
# ----------------------------------------------------------------------
@dataclass
class GenerationResult:
    """What the UI actually needs after asking a question."""

    answer: str
    sources: list[RetrievedChunk]
    prompt_used: str  # exposed for Debug Mode


class Generator:
    """
    Combines retrieved chunks + chat history into a grounded prompt,
    calls the LLM, and returns the answer alongside its sources.
    """

    def __init__(self, provider: BaseLLMProvider | None = None):
        self.provider = provider or get_llm_provider()

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        chat_history: list[tuple[str, str]] | None = None,
    ) -> GenerationResult:
        """
        Produce a grounded answer to `query` using `retrieved_chunks`
        as the only source of truth.

        If retrieved_chunks is empty, we still call the LLM (with an
        explicit "no context found" note in the prompt) so it responds
        with the required "not enough information" message rather than
        guessing from its own parametric knowledge.
        """
        context_dicts = [
            {"text": c.text, "source": c.source, "page": c.page, "chunk_id": c.chunk_id}
            for c in retrieved_chunks
        ]
        prompt = build_rag_prompt(query, context_dicts, chat_history)

        answer = self.provider.complete(SYSTEM_PROMPT, prompt)

        return GenerationResult(
            answer=answer,
            sources=retrieved_chunks,
            prompt_used=prompt,
        )
