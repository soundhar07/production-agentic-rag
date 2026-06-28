"""
One-time ingestion CLI for the Constitution PDF.

    uv run python -m app.ingest            # full build (embeds -> Chroma + chunks.jsonl)
    uv run python -m app.ingest --dry-run  # parse + report only, no embedding/API calls

Loads ``docs/constitution.pdf``, tags each chunk with ``{article, part, page}``,
embeds with a local Ollama model (``qwen3-embedding:4b``), and persists a Chroma
vector store plus ``chunks.jsonl`` (used to rebuild BM25 at API startup). The
corpus is static, so this runs once.

Headings are detected with line-leading regexes tuned to this PDF: Article bodies
start like ``21. Right ... .—<text>``, Parts are bare ``PART III`` lines, and
Schedules/Appendices reset article tracking. Amendment footnotes are left in the
text (minor noise) — the only footnote handling kept is a guard that stops a
footnote line like ``1. Subs. by ...`` from being misread as an Article heading.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import get_settings

# Article heading: number + optional letters (21, 21A, 243ZH) + dot + title.
ARTICLE_RE = re.compile(r"^\s*(\d+[A-Z]*)\.\s+\S")
# Strip a leading superscript footnote ref like "3[" before testing for a heading.
LEADING_REF_RE = re.compile(r"^\s*\d+\[")
# Part heading on its own line: "PART III", "PART IXA".
PART_RE = re.compile(r"^\s*PART\s+([IVXLC]+[A-Z]*)\s*$")
# Schedule / Appendix heading — all-caps "<ORDINAL> SCHEDULE" (optionally "THE ..."
# or prefixed by a footnote ref like "1["), e.g. "1[FOURTH SCHEDULE", or "APPENDIX II".
SCHEDULE_RE = re.compile(
    r"^\s*(?:\d+\[)?(?:THE\s+)?"
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)"
    r"\s+SCHEDULE\b|^\s*(?:\d+\[)?APPENDIX\s+[IVXLC]+\b"
)
# A footnote line ("1. Subs. by the Constitution ... Act ...") shares the "N." shape
# of an Article heading; this guard keeps it from being tagged as an Article.
AMENDMENT_KEYWORDS = re.compile(
    r"(Subs\.\s*by|Ins\.\s*by|Added\s*by|Omitted\s*by|omitted\s*by|Rep\.\s*by|"
    r"w\.e\.f\.|\bibid\b|Amendment\)?\s*Act)",
    re.IGNORECASE,
)


def clean_page(text: str) -> list[str]:
    """Return the non-blank lines of one page."""
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()]


def article_heading(line: str) -> str | None:
    """Return the article number if `line` starts a new article body, else None."""
    stripped = LEADING_REF_RE.sub("", line)
    if AMENDMENT_KEYWORDS.search(stripped[:80]):  # footnote, not a heading
        return None
    m = ARTICLE_RE.match(stripped)
    return m.group(1) if m else None


def build_documents(settings) -> list[Document]:
    """Load the PDF and produce article/part/page-tagged chunks."""
    pages = PyPDFLoader(settings.pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    cur_part: str | None = None
    cur_article: str | None = None
    in_tail = False  # True once we enter Schedules/Appendices (no article numbers there)
    docs: list[Document] = []

    for page in pages:
        page_idx = page.metadata.get("page", 0)  # 0-based PyPDFLoader index
        if page_idx < settings.body_start_page:
            continue  # skip cover / preface / table of contents
        page_no = page_idx + 1  # human-facing page number

        # Split the page into segments, each tagged with the part/article active
        # at its position; a new heading starts a new segment.
        segments: list[tuple[str | None, str | None, list[str]]] = []
        buf: list[str] = []

        def flush() -> None:
            if buf:
                segments.append((cur_part, cur_article, buf.copy()))
                buf.clear()

        for ln in clean_page(page.page_content):
            part_m = PART_RE.match(ln)
            if part_m:
                flush()
                cur_part = f"Part {part_m.group(1)}"
                cur_article = None
                in_tail = False
                continue  # drop the bare "PART III" line
            if SCHEDULE_RE.match(ln):
                flush()
                cur_part = LEADING_REF_RE.sub("", ln).strip()  # e.g. "SEVENTH SCHEDULE"
                cur_article = None
                in_tail = True
                buf.append(ln)
                continue
            if not in_tail:  # inside Schedules, numbered entries are not articles
                art = article_heading(ln)
                if art:
                    flush()
                    cur_article = art
            buf.append(ln)
        flush()

        for part, article, seg_lines in segments:
            seg_text = "\n".join(seg_lines).strip()
            if not seg_text:
                continue
            for chunk in splitter.split_text(seg_text):
                chunk = chunk.strip()
                if len(chunk) < 30:  # drop tiny fragments
                    continue
                docs.append(
                    Document(
                        page_content=chunk,
                        # Chroma rejects None metadata values -> use "" for absent.
                        metadata={"page": page_no, "part": part or "", "article": article or ""},
                    )
                )
    return docs


def _write_chunks_jsonl(docs: list[Document], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"page_content": d.page_content, "metadata": d.metadata}) + "\n")


def _embed_and_persist(docs: list[Document], settings) -> None:
    """Embed all chunks into a persisted Chroma collection using local Ollama."""
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma

    print(f"Embedding {len(docs)} chunks with {settings.embedding_model}...")
    Chroma.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model=settings.embedding_model),
        persist_directory=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )


def _report(docs: list[Document]) -> None:
    with_article = sum(1 for d in docs if d.metadata["article"])
    print(f"\nChunks: {len(docs)}  |  with article tag: {with_article}  "
          f"|  page-only (Preamble/Schedule): {len(docs) - with_article}")
    print("--- sample chunks ---")
    for d in docs[:3]:
        tag = f"art={d.metadata['article'] or '-'} part={d.metadata['part'] or '-'} p={d.metadata['page']}"
        print(f"[{tag}] {d.page_content[:80]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Constitution PDF.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse + report only; no embedding or API calls.")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Loading {settings.pdf_path} (body starts at page index {settings.body_start_page})...")
    docs = build_documents(settings)
    _report(docs)

    if args.dry_run:
        print("\n[dry-run] No embedding performed.")
        return

    # Rebuild from scratch so re-runs aren't appended/duplicated.
    persist = Path(settings.chroma_persist_dir)
    if persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {settings.chunks_path} (for BM25 rebuild at startup)...")
    _write_chunks_jsonl(docs, settings.chunks_path)

    _embed_and_persist(docs, settings)
    print(f"\nDone. Persisted Chroma collection '{settings.chroma_collection}' "
          f"to {settings.chroma_persist_dir}.")


if __name__ == "__main__":
    main()
