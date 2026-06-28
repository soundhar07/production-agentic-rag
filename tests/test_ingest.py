"""
Tests for the ingest parser (article/part metadata + line cleaning).

Deterministic and free — no LLM calls, no embeddings, no PDF. Exercises the pure
text-processing helpers on small inline samples that mirror the real PDF's quirks.
"""

from app.ingest import clean_page, article_heading


class TestArticleHeading:
    def test_simple_article_heading(self):
        assert article_heading("21. Protection of life and personal liberty.—No person") == "21"

    def test_article_with_letter_suffix(self):
        assert article_heading("243ZH. Definitions.—In this Part") == "243ZH"

    def test_superscript_ref_prefix_stripped(self):
        # "3[2A. ..." — a footnote-ref digit glued before the real article number.
        # This particular line is an *omitted* article (amendment text) -> not a heading.
        assert article_heading("3[2A. [Sikkim].—Omitted by the Constitution") is None

    def test_footnote_line_is_not_a_heading(self):
        # Collides with article "1." but carries amendment keywords -> footnote.
        # This guard is what keeps footnotes from corrupting Article tagging.
        assert article_heading("1. Subs. by the Constitution (Forty-second Amendment) Act") is None

    def test_continuation_line_is_not_a_heading(self):
        assert article_heading("shall be a Union of States.") is None

    def test_clause_line_is_not_a_heading(self):
        assert article_heading("(3) The territory of India shall comprise—") is None


class TestCleanPage:
    def test_returns_nonblank_lines(self):
        page = "4. Laws made under articles 2 and 3.—Any law\n\n   \nshall apply.\n"
        assert clean_page(page) == [
            "4. Laws made under articles 2 and 3.—Any law",
            "shall apply.",
        ]

    def test_keeps_headers_and_footnotes(self):
        # The simplified parser no longer strips these — they remain as minor noise.
        page = "3 THE CONSTITUTION OF INDIA\n1. Subs. by the Constitution Act"
        kept = clean_page(page)
        assert "3 THE CONSTITUTION OF INDIA" in kept
        assert any("Subs. by" in ln for ln in kept)
