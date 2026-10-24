"""Run pytest and write the test report requested by the unit brief."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs"


def main():
    """Write human-readable and JUnit XML reports; propagate pytest exit code."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "tests", "-ra", "-q", f"--junitxml={REPORT_DIR / 'test_report.xml'}"]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    output = result.stdout + ("\n" + result.stderr if result.stderr else "")
    (REPORT_DIR / "test_summary.txt").write_text(output, encoding="utf-8")
    print(output)
    print(f"Test summary: {REPORT_DIR / 'test_summary.txt'}")
    print(f"JUnit report: {REPORT_DIR / 'test_report.xml'}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
