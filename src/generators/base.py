from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseAnswerGenerator(ABC):
    """Common interface for local or remote answer generators."""

    model_id: str
    provider: str

    @abstractmethod
    def generate_answer(
        self,
        query: str,
        evidence_chunks: List[Any],
        instruction: Optional[str] = None,
    ) -> str:
        """Generate one evidence-grounded answer."""
