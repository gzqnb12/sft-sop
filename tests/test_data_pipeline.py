import tempfile
import unittest
from pathlib import Path

from sft_sop.build_data import build_records, write_jsonl
from sft_sop.check_data import validate_dataset


class DataPipelineTest(unittest.TestCase):
    def test_dataset_counts_and_validation(self) -> None:
        records = build_records()
        self.assertEqual(
            {split: len(rows) for split, rows in records.items()},
            {"train": 60, "validation": 15, "test": 15},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            for split, rows in records.items():
                write_jsonl(data_dir / f"{split}.jsonl", rows)

            self.assertEqual(
                validate_dataset(data_dir),
                {"train": 60, "validation": 15, "test": 15},
            )

    def test_each_split_has_distinct_user_prompts(self) -> None:
        records = build_records()
        prompt_sets = {
            split: {row["messages"][1]["content"] for row in rows}
            for split, rows in records.items()
        }
        self.assertTrue(prompt_sets["train"].isdisjoint(prompt_sets["validation"]))
        self.assertTrue(prompt_sets["train"].isdisjoint(prompt_sets["test"]))
        self.assertTrue(prompt_sets["validation"].isdisjoint(prompt_sets["test"]))


if __name__ == "__main__":
    unittest.main()
