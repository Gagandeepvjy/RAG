"""Stage 1: Query expansion with HyDE (Hypothetical Document Embeddings).

Primary strategy — Multi-query via Ollama:
  Generate N semantically varied reformulations of the user query.

HyDE fallback (when Ollama is unavailable):
  Ask the LLM to write a short hypothetical answer, then use that text as an
  additional query embedding. This often outperforms pure multi-query because
  the hypothetical answer lives in document space rather than query space.

If Ollama is completely unavailable, falls back gracefully to the original
query only so the rest of the pipeline always runs.
"""

import ollama

from src.config import settings


class QueryExpander:
    """Generates query variants using multi-query expansion + HyDE."""

    EXPANSION_SYSTEM_PROMPT = (
        "You are a search query expansion assistant. Given a user question, "
        "generate alternative phrasings that capture the same intent using "
        "different vocabulary and sentence structures. Return ONLY the queries, "
        "one per line, with no numbering, bullets, or extra text."
    )

    HYDE_SYSTEM_PROMPT = (
        "You are a helpful assistant. Given a question, write a short, dense, "
        "factual paragraph that directly answers it. Write as if you are "
        "authoring a section of a document. Do not explain that you are "
        "generating a hypothetical — just write the answer paragraph."
    )

    def __init__(self):
        self.client = ollama.Client(host=settings.ollama_url or None)
        self.model = settings.ollama_expansion_model

    def expand(self, query: str, num_expansions: int | None = None) -> list[str]:
        """
        Returns the original query + multi-query variants + HyDE document.
        Falls back gracefully to [query] if Ollama is unavailable.
        """
        n = num_expansions or settings.num_query_expansions
        queries = [query]

        try:
            # --- Multi-query expansion ---
            expansion_response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.EXPANSION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Original question:\n{query}\n\n"
                            f"Generate {n} alternative search queries:"
                        ),
                    },
                ],
                options={"temperature": 0.7},
            )
            raw = expansion_response.message.content or ""
            for line in raw.strip().split("\n"):
                line = line.strip()
                if line and line.lower() != query.lower() and line not in queries:
                    queries.append(line)

            # --- HyDE: hypothetical answer as an additional query ---
            hyde_response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.HYDE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                options={"temperature": 0.3},
            )
            hyde_doc = (hyde_response.message.content or "").strip()
            if hyde_doc and hyde_doc not in queries:
                queries.append(hyde_doc)

        except Exception:
            pass  # Ollama unavailable — original query is enough to continue

        return queries[: n + 2]  # original + n expansions + 1 HyDE
