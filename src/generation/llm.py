"""Stage 7: Answer generation with streaming, citations, and conversation history."""

from collections import deque

import ollama

from src.config import settings
from src.models import DocumentChunk

# Each history entry: {"role": "user"|"assistant", "content": str}
ConversationHistory = deque


class AnswerGenerator:
    """Synthesizes a grounded, cited answer from retrieved chunks with streaming."""

    SYSTEM_PROMPT = (
        "You are a precise, helpful assistant that answers questions using ONLY "
        "the numbered context passages provided. Rules:\n"
        "1. Cite every factual claim with its passage number, e.g. [1], [2].\n"
        "2. If multiple passages support a claim, cite all of them, e.g. [1][3].\n"
        "3. If the context does not contain enough information, say exactly: "
        "'The provided documents do not contain enough information to answer this question.'\n"
        "4. Do NOT use outside knowledge. Do NOT make up citations.\n"
        "5. Be concise and direct."
    )

    def __init__(self):
        self.client = ollama.Client(host=settings.ollama_url or None)
        self.model = settings.ollama_chat_model
        self.history: ConversationHistory = deque(
            maxlen=settings.conversation_history_len * 2  # user+assistant pairs
        )

    def generate(self, query: str, chunks: list[DocumentChunk]) -> str:
        if not chunks:
            return "No relevant context found for this question."

        context = self._format_context(chunks)
        user_message = f"Context passages:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *list(self.history),
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.1},
            )
            full_response = response.message.content or ""

            # Save turn to history
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": full_response})

            return full_response

        except Exception as e:
            return f"(LLM generation failed: {e})"

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.history.clear()

    @staticmethod
    def _format_context(chunks: list[DocumentChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.source.split("/")[-1]
            parts.append(f"[{i}] (source: {source})\n{chunk.text}")
        return "\n\n---\n\n".join(parts)
