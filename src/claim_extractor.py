import re
from typing import List


def extract_claims(answer: str) -> List[str]:
    """
    Extract simple claims from the generated answer.

    Valid sentences are decomposed into atomic claims when they contain a
    recognizable list. Simple sentences remain unchanged.
    """

    if not answer or not answer.strip():
        return []

    cleaned_answer = answer.strip()

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_answer)

    claims = []

    for sentence in sentences:
        sentence = clean_claim(sentence)

        for atomic_claim in split_compound_claim(sentence):
            if not is_valid_claim(atomic_claim):
                continue
            claims.append(atomic_claim)

    return claims


def _split_list_items(text: str) -> List[str]:
    """Split a comma-delimited noun-phrase list without splitting inner prose."""
    text = text.strip().rstrip(".!?")
    if not text:
        return []

    if "," in text:
        parts = re.split(r"\s*,\s*", text)
        if parts:
            parts[-1] = re.sub(r"^(?:and|or)\s+", "", parts[-1], flags=re.I)
        return [part.strip() for part in parts if part.strip()]

    # A two-item list is safe to split only when both sides are short phrases.
    parts = re.split(r"\s+(?:and|or)\s+", text, maxsplit=1, flags=re.I)
    if len(parts) == 2 and all(0 < len(part.split()) <= 8 for part in parts):
        return [part.strip() for part in parts]
    return [text]


def _subject_has(subject: str) -> str:
    return "have" if re.search(r"(?:systems|models|documents|they)$", subject, re.I) else "has"


def _risk_item_claim(subject: str, item: str) -> str:
    """Rebuild one atomic risk claim while retaining the supplied meaning."""
    item = re.sub(r"^risks?\s+from\s+", "", item.strip(), flags=re.I)
    issue = re.fullmatch(r"(.+?)\s+(?:issues?|concerns?)", item, flags=re.I)
    verb = _subject_has(subject)
    if issue:
        return f"{subject} {verb} {issue.group(1).strip()} risks."
    return f"{subject} {verb} risks from {item}."


def split_compound_claim(sentence: str) -> List[str]:
    """Split supported list-style constructions into independently verifiable claims."""
    sentence = clean_claim(sentence)
    if not sentence:
        return []
    body = sentence.rstrip(".!?")

    # Example: "Systems have risks from A, B, and C."
    direct_risks = re.match(
        r"^(?P<subject>.+?)\s+(?:have|has)\s+risks?\s+from\s+(?P<items>.+)$",
        body,
        flags=re.I,
    )
    if direct_risks:
        items = _split_list_items(direct_risks.group("items"))
        if len(items) > 1:
            subject = direct_risks.group("subject").strip()
            return [_risk_item_claim(subject, item) for item in items]

    # For general "including" / "such as" lists, preserve the original lead-in
    # and attach only one list item to each atomic claim.
    example_list = re.match(
        r"^(?P<lead>.+?)\s+(?P<marker>including|such as)\s+(?P<items>.+)$",
        body,
        flags=re.I,
    )
    if example_list:
        items = _split_list_items(example_list.group("items"))
        if len(items) > 1:
            lead = example_list.group("lead").rstrip(" ,")
            marker = example_list.group("marker")
            first_has_risk_prefix = bool(
                re.match(r"risks?\s+from\s+", items[0], flags=re.I)
            )
            atomic_claims = []
            for item in items:
                if first_has_risk_prefix and not re.match(
                    r"(?:risks?\s+from|potential\s+negative\s+impacts?)",
                    item,
                    flags=re.I,
                ):
                    item = f"risks from {item}"
                atomic_claims.append(f"{lead} {marker} {item}.")
            return atomic_claims

    return [sentence]


def clean_claim(sentence: str) -> str:
    """
    Clean a claim sentence.
    """

    sentence = sentence.strip()
    sentence = sentence.strip("\"'“”‘’")
    sentence = sentence.strip()
    return sentence


def is_valid_claim(sentence: str) -> bool:
    """
    Check whether a sentence looks like a useful claim.
    """

    if not sentence:
        return False

    words = sentence.split()

    if len(words) < 5:
        return False

    alphabetic_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence)

    if len(alphabetic_words) < 4:
        return False

    if sentence.count(",") >= 3 and len(words) < 15:
        return False

    return True


def print_claims(claims: List[str]) -> None:
    """
    Print extracted claims in terminal.
    """

    print("\nExtracted Claims")
    print("=" * 70)

    if not claims:
        print("No valid claims were extracted.")
        return

    for index, claim in enumerate(claims, start=1):
        print(f"{index}. {claim}")
