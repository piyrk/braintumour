from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def check_exists(path: Path, label: str) -> CheckResult:
    exists = path.exists()
    return CheckResult(label, exists, f"{path} {'found' if exists else 'missing'}")


def check_glob_count(folder: Path, pattern: str, minimum: int, label: str) -> CheckResult:
    count = len(list(folder.rglob(pattern))) if folder.exists() else 0
    passed = count >= minimum
    return CheckResult(label, passed, f"{count} file(s) matched {pattern} in {folder}")


def run_checks(require_training_artifacts: bool = True):
    checks = [
        check_exists(Path("train_detection.py"), "Detection training script"),
        check_exists(Path("train_segmentation.py"), "Segmentation training script"),
        check_exists(Path("train_classifier.py"), "Classifier training script"),
        check_exists(Path("train_gan.py"), "GAN training script"),
        check_exists(Path("streamlit_app.py"), "Streamlit app"),
        check_exists(Path("preprocess_data.py"), "Preprocessing script"),
        check_exists(Path("docs/architecture_diagram.md"), "Architecture diagram doc"),
        check_exists(Path("docs/report_template.md"), "Report template"),
        check_exists(Path("docs/submission_checklist.md"), "Submission checklist"),
        check_exists(Path("build_submission_package.py"), "Submission package builder"),
    ]

    if require_training_artifacts:
        checks.extend(
            [
                check_glob_count(Path("artifacts"), "*.keras", 1, "At least one model checkpoint"),
                check_glob_count(Path("artifacts"), "*.csv", 1, "At least one training log"),
                check_glob_count(Path("artifacts"), "*.png", 1, "At least one metric/sample image"),
            ]
        )

    return checks


def parse_args():
    parser = argparse.ArgumentParser(description="Validate project submission readiness.")
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Skip checks that require trained artifacts such as checkpoints and logs.",
    )
    return parser.parse_args()


def print_results(results: list[CheckResult]):
    print("Submission validation report")
    print("=" * 32)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.details}")

    failures = [result for result in results if not result.passed]
    print("=" * 32)
    if failures:
        print(f"Validation failed with {len(failures)} issue(s).")
        return 1
    print("Validation passed.")
    return 0


def main():
    args = parse_args()
    results = run_checks(require_training_artifacts=not args.no_training)
    code = print_results(results)
    sys.exit(code)


if __name__ == "__main__":
    main()
