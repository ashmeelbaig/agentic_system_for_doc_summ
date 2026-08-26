import re
from typing import List


STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
    "and", "or", "how", "does", "do", "these", "this", "that", "with",
    "from", "by", "as", "it", "be", "was", "were", "about", "according",
    "explain", "describe", "discuss", "main"
}


def extract_query_keywords(query: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", query.lower())

    keywords = []
    for word in words:
        if word not in STOPWORDS and word not in keywords:
            keywords.append(word)

    return keywords


def rewrite_query(query: str, max_terms: int = 12) -> str:
    """
    Generic and efficient query rewrite.

    This does not use hard-coded domain vocabulary.
    It keeps the original query and adds cleaned important terms
    from the question itself.
    """

    keywords = extract_query_keywords(query)

    if not keywords:
        return query

    keywords = keywords[:max_terms]

    rewritten_query = query + " " + " ".join(keywords)

    return rewritten_query