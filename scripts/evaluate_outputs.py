from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation import summarize_result_file


OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    if not OUTPUT_DIR.exists():
        print("No outputs folder found.")
        return

    result_files = sorted(OUTPUT_DIR.glob("*.json"))

    if not result_files:
        print("No JSON result files found in outputs folder.")
        return

    print("\nEvaluation Summary")
    print("=" * 80)

    for result_file in result_files:
        summary = summarize_result_file(result_file)

        print(f"\nFile: {summary['file_name']}")
        print(f"PDF: {summary['pdf_name']}")
        print(
            f"Generator: {summary['generator_mode']} / "
            f"{summary['generator_model']}"
        )
        print(f"Retrieved evidence: {summary['retrieved_evidence_count']}")
        print(f"Total claims: {summary['total_claims']}")
        print(f"Supported claims: {summary['supported_claims']}")
        print(f"Partially supported claims: {summary['partially_supported_claims']}")
        print(f"Unsupported claims: {summary['unsupported_claims']}")
        print(f"Contradicted claims: {summary['contradicted_claims']}")
        print(f"Not enough evidence claims: {summary['not_enough_evidence_claims']}")
        print(f"Faithfulness score: {summary['faithfulness_score']:.2f}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()