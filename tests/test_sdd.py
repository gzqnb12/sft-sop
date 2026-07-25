import tempfile
import unittest
from pathlib import Path

from sft_sop.check_sdd import validate_specs


class SddValidationTest(unittest.TestCase):
    def test_repository_specs_are_valid(self) -> None:
        specs_dir = Path(__file__).resolve().parents[1] / "specs"
        self.assertEqual(validate_specs(specs_dir), [])

    def test_missing_traceability_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            specs_dir = Path(temp_dir)
            (specs_dir / "README.md").write_text("# Workflow\n", encoding="utf-8")
            feature_dir = specs_dir / "002-example"
            feature_dir.mkdir()
            (feature_dir / "spec.md").write_text(
                "# Example\n\nStatus: Accepted\n\n"
                "## Functional requirements\n\n- FR-001: Work.\n\n"
                "## Acceptance criteria\n\n- AC-001: It works.\n",
                encoding="utf-8",
            )
            (feature_dir / "plan.md").write_text(
                "# Plan\n\nStatus: Accepted\n\n## Verification\n\n- Run tests.\n",
                encoding="utf-8",
            )
            (feature_dir / "tasks.md").write_text(
                "# Tasks\n\nStatus: Accepted\n\n- [ ] T-001 Do work.\n",
                encoding="utf-8",
            )

            errors = validate_specs(specs_dir)
            self.assertTrue(any("FR-001 is not traced" in error for error in errors))
            self.assertTrue(any("AC-001 is not traced" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
