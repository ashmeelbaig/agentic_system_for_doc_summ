import re
from typing import Dict, List


SUSPICIOUS_PATTERNS = (
    "ignore previous instructions", "ignore above instructions",
    "disregard instructions", "reveal system prompt", "show hidden prompt",
    "developer message", "system message", "jailbreak", "do anything now",
    "override safety", "follow only this instruction", "delete files",
    "execute command", "run shell", "install malware", "exfiltrate",
    "send credentials", "api key", "password",
)

EVIDENCE_WARNING = "[Potential prompt injection text treated as document content only]"


def _find_patterns(text: str) -> List[str]:
    return [
        pattern for pattern in SUSPICIOUS_PATTERNS
        if re.search(re.escape(pattern), text or "", flags=re.IGNORECASE)
    ]


def detect_prompt_injection(text: str) -> Dict[str, object]:
    """Detect obvious prompt-injection or unsafe instruction phrases."""
    matched_patterns = _find_patterns(text)
    is_suspicious = bool(matched_patterns)
    return {
        "is_suspicious": is_suspicious,
        "matched_patterns": matched_patterns,
        "reason": (
            "Potential prompt injection or unsafe instruction text was detected."
            if is_suspicious else "No suspicious instruction patterns were detected."
        ),
    }


def sanitize_evidence_text(text: str) -> str:
    """Mark suspicious evidence as inert document content without deleting it."""
    if not detect_prompt_injection(text)["is_suspicious"]:
        return text
    if text.startswith(EVIDENCE_WARNING):
        return text
    return f"{EVIDENCE_WARNING}\n{text}"


def check_user_query_safety(query: str) -> Dict[str, object]:
    """Reject queries that request prompt extraction or unsafe instructions."""
    detection = detect_prompt_injection(query)
    is_safe = not detection["is_suspicious"]
    return {
        "is_safe": is_safe,
        "reason": (
            "The query asks for unsafe or non-document behaviour."
            if not is_safe else "No unsafe query patterns were detected."
        ),
        "matched_patterns": detection["matched_patterns"],
    }
