from pathlib import Path
import sys

# Add project root to Python path so "src" can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import summarize_result_file, save_evaluation_summary_csv


OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    if not OUTPUT_DIR.exists():
        print("No outputs folder found.")
        return

    result_files = sorted(OUTPUT_DIR.glob("*.json"))

    # Avoid reading the CSV export or any non-result JSON later if added
    result_files = [
        file for file in result_files
        if file.name.startswith("result_")
    ]

    if not result_files:
        print("No JSON result files found in outputs folder.")
        return

    summaries = []

    print("\nEvaluation Summary")
    print("=" * 80)

    for result_file in result_files:
        summary = summarize_result_file(result_file)
        summaries.append(summary)

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

    csv_path = OUTPUT_DIR / "evaluation_summary.csv"

    save_evaluation_summary_csv(
        summaries=summaries,
        output_csv_path=csv_path,
    )

    print(f"\nCSV summary saved to: {csv_path}")
    print("Evaluation completed.")


if __name__ == "__main__":
    main()