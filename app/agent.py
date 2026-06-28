"""
Agentic RAG agent (LangGraph) over the Constitution PDF.

Flow:  retrieve -> grade -> [rewrite -> retrieve]* -> generate
                                   |
                                   +-> (no survivors, out of retries) -> fallback

All reasoning steps (grade, rewrite, generate) run on the primary Gemini model
(see docs/adr/0002). Ollama is only a generate-time fallback if Gemini errors.
Grading is binary (relevant / not); routing keys off how many docs survive.
"""

from __future__ import annotations

import re
from typing import Optional, Literal

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langsmith import traceable

from app.config import get_settings
from app.retriever import build_retriever


# === State ===

class RAGState(TypedDict):
    query: str
    rewritten_query: str
    documents: list[Document]
    generation: str
    retry_count: int
    model_used: str
    error: Optional[str]
    sources: list[dict]


def _extract_text(content) -> str:
    """Normalise an LLM `.content` to a string.

    Gemini (`gemini-3.1-flash-lite`) returns a list of content-block dicts
    (`[{"type": "text", "text": "..."}]`); Ollama returns a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _format_docs(docs: list[Document]) -> str:
    out = []
    for i, d in enumerate(docs):
        art = d.metadata.get("article") or "-"
        page = d.metadata.get("page", "?")
        out.append(f"[{i}] (Article {art}, p. {page})\n{d.page_content}")
    return "\n\n".join(out)


def _make_sources(docs: list[Document]) -> list[dict]:
    """Deduped, sorted citation list from the metadata of the given docs."""
    seen = set()
    sources = []
    for d in docs:
        article = d.metadata.get("article") or None
        part = d.metadata.get("part") or None
        page = d.metadata.get("page")
        key = (article, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"article": article, "page": page, "part": part})
    # Sort: page-numbered first by page, articles grouped; stable + readable.
    sources.sort(key=lambda s: (s["page"] is None, s["page"] or 0))
    return sources


# --- LangSmith span shaping -------------------------------------------------
# Node methods receive (self, state); without shaping, traces would log the whole
# agent object and every Document. These keep spans small and meaningful.

def _node_inputs(inputs: dict) -> dict:
    state = inputs.get("state", {})
    return {
        "query": state.get("query"),
        "rewritten_query": state.get("rewritten_query") or None,
        "retry_count": state.get("retry_count", 0),
        "num_documents": len(state.get("documents", []) or []),
    }


def _node_outputs(output) -> dict:
    if not isinstance(output, dict):
        return {"output": output}
    summary: dict = {}
    if "documents" in output:
        docs = output["documents"]
        summary["num_documents"] = len(docs)
        summary["articles"] = sorted(
            {d.metadata.get("article") for d in docs if d.metadata.get("article")}
        )
    for key in ("rewritten_query", "generation", "model_used", "sources"):
        if key in output:
            summary[key] = output[key]
    return summary


class ProductionAgent:
    """Agentic RAG over the Constitution, with graceful model fallback on generate."""

    def __init__(self):
        settings = get_settings()
        self.primary_llm = ChatGoogleGenerativeAI(model=settings.primary_model, temperature=0)
        self.fallback_llm = ChatOllama(model=settings.fallback_model, temperature=0)
        self.max_rag_retries = settings.max_rag_retries
        self.retriever = build_retriever()
        self.graph = self._build_graph()

    # --- Nodes ---

    @traceable(name="rag.retrieve", run_type="retriever",
               process_inputs=_node_inputs, process_outputs=_node_outputs)
    def _retrieve(self, state: RAGState) -> dict:
        query = state.get("rewritten_query") or state["query"]
        docs = self.retriever.invoke(query)
        print(f"[RETRIEVE] '{query[:60]}' -> {len(docs)} docs")
        return {"documents": docs}

    @traceable(name="rag.grade", run_type="chain",
               process_inputs=_node_inputs, process_outputs=_node_outputs)
    def _grade(self, state: RAGState) -> dict:
        docs = state["documents"]
        if not docs:
            return {"documents": []}

        excerpts = "\n\n".join(
            f"[{i}] {d.page_content[:500]}" for i, d in enumerate(docs)
        )
        prompt = (
            "You grade retrieved excerpts from the Constitution of India for relevance "
            "to a question.\n\n"
            f"Question: {state['query']}\n\n"
            f"Excerpts:\n{excerpts}\n\n"
            "Return ONLY the indices of the excerpts that contain information useful to "
            'answer the question, as a comma-separated list (e.g. "0, 2"). '
            'If none are relevant, return "NONE".'
        )
        try:
            raw = _extract_text(self.primary_llm.invoke(prompt).content).strip()
        except Exception as e:  # noqa: BLE001 - fail open, never starve generate
            print(f"[GRADE] error ({e}); keeping all docs")
            return {"documents": docs}

        if "NONE" in raw.upper() and not re.search(r"\d", raw):
            print("[GRADE] no relevant docs")
            return {"documents": []}

        idxs = {int(n) for n in re.findall(r"\d+", raw) if int(n) < len(docs)}
        if not idxs:  # unparseable -> fail open
            print(f"[GRADE] unparseable reply {raw!r}; keeping all docs")
            return {"documents": docs}

        survivors = [docs[i] for i in sorted(idxs)]
        print(f"[GRADE] kept {len(survivors)}/{len(docs)} docs")
        return {"documents": survivors}

    @traceable(name="rag.rewrite", run_type="chain",
               process_inputs=_node_inputs, process_outputs=_node_outputs)
    def _rewrite(self, state: RAGState) -> dict:
        prompt = (
            "The following question did not retrieve relevant excerpts from the "
            "Constitution of India. Rewrite it to be more specific and use terms likely "
            "to appear in the constitutional text (article numbers, legal terms). "
            "Return ONLY the rewritten question.\n\n"
            f"Question: {state['query']}"
        )
        try:
            new_q = _extract_text(self.primary_llm.invoke(prompt).content).strip()
        except Exception:  # noqa: BLE001 - degrade to original query
            new_q = state["query"]
        print(f"[REWRITE] -> '{new_q[:60]}'")
        return {"rewritten_query": new_q, "retry_count": state.get("retry_count", 0) + 1}

    @traceable(name="rag.generate", run_type="chain",
               process_inputs=_node_inputs, process_outputs=_node_outputs)
    def _generate(self, state: RAGState) -> dict:
        docs = state["documents"]
        prompt = (
            "You answer questions about the Constitution of India using ONLY the "
            "provided excerpts. If the excerpts do not contain the answer, say you could "
            "not find it in the provided text. Be concise and accurate.\n\n"
            f"Excerpts:\n{_format_docs(docs)}\n\n"
            f"Question: {state['query']}\nAnswer:"
        )
        sources = _make_sources(docs)
        try:
            answer = _extract_text(self.primary_llm.invoke(prompt).content)
            return {"generation": answer, "model_used": "primary", "error": None,
                    "sources": sources}
        except Exception as primary_err:  # noqa: BLE001
            print(f"[GENERATE] primary failed ({primary_err}); trying fallback")
            try:
                answer = _extract_text(self.fallback_llm.invoke(prompt).content)
                return {"generation": answer, "model_used": "fallback", "error": None,
                        "sources": sources}
            except Exception as fb_err:  # noqa: BLE001
                return {
                    "generation": ("I'm sorry, I'm having trouble answering right now. "
                                   "Please try again in a moment."),
                    "model_used": "error_handler",
                    "error": str(fb_err),
                    "sources": [],
                }

    @traceable(name="rag.fallback", run_type="chain",
               process_inputs=_node_inputs, process_outputs=_node_outputs)
    def _fallback(self, state: RAGState) -> dict:
        print("[FALLBACK] no relevant constitutional provisions found")
        return {
            "generation": ("I couldn't find relevant provisions in the Constitution to "
                           "answer that. Try rephrasing, or asking about a specific "
                           "Article or topic."),
            "model_used": "fallback",
            "error": None,
            "sources": [],
        }

    def _route_after_grade(self, state: RAGState) -> Literal["generate", "rewrite", "fallback"]:
        survivors = len(state["documents"])
        retries = state.get("retry_count", 0)
        if survivors >= 1:
            return "generate"
        if retries < self.max_rag_retries:
            print(f"[ROUTER] 0 survivors, retry {retries + 1}/{self.max_rag_retries} -> rewrite")
            return "rewrite"
        return "fallback"

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)
        g.add_node("fallback", self._fallback)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", END)
        g.add_edge("fallback", END)
        return g.compile()

    @traceable(name="agentic_rag_invoke")
    def invoke(self, message: str) -> dict:
        """Run the RAG loop. Returns {response: str, model_used, error, sources}."""
        result = self.graph.invoke({
            "query": message,
            "rewritten_query": "",
            "documents": [],
            "generation": "",
            "retry_count": 0,
            "model_used": "",
            "error": None,
            "sources": [],
        })
        return {
            "response": result["generation"],
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
            "sources": result.get("sources", []),
            # Text of the surviving retrieved passages — used by RAGAS evaluation
            # (eval/ragas_eval.py); the request pipeline ignores this key.
            "contexts": [d.page_content for d in result.get("documents", [])],
        }
