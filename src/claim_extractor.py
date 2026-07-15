import re
from typing import List


def extract_claims(answer: str) -> List[str]:
    """
    Extract simple claims from the generated answer.

    For the current prototype, each valid sentence is treated as one claim.
    """

    if not answer or not answer.strip():
        return []

    cleaned_answer = answer.strip()

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_answer)

    claims = []

    for sentence in sentences:
        sentence = clean_claim(sentence)

        if not is_valid_claim(sentence):
            continue

        claims.append(sentence)

    return claims


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

    if len(words) < 6:
        return False

    alphabetic_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence)

    if len(alphabetic_words) < 5:
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