from src.query_rewriter import rewrite_query


def test_rewrite_keeps_original_query_and_important_keywords():
    query = "How does photosynthesis use chlorophyll?"

    rewritten = rewrite_query(query)

    assert rewritten.startswith(query)
    assert "photosynthesis" in rewritten.lower()
    assert "chlorophyll" in rewritten.lower()


def test_empty_query_returns_empty_string():
    assert rewrite_query("") == ""


def test_rewrite_uses_keywords_from_arbitrary_query_not_domain_vocabulary():
    query = "Explain zeugma in classical rhetoric"

    rewritten = rewrite_query(query)

    assert rewritten.startswith(query)
    assert "zeugma" in rewritten.lower()
    assert "rhetoric" in rewritten.lower()
