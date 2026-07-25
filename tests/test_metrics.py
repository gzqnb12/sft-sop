import unittest

from sft_sop.metrics import compute_metrics, is_valid_schema, parse_json_output


class MetricsTest(unittest.TestCase):
    def test_parse_json_output_accepts_markdown_wrapper(self) -> None:
        parsed = parse_json_output(
            '说明如下：\n```json\n{"intent":"refund","urgency":"high"}\n```'
        )
        self.assertEqual(parsed, {"intent": "refund", "urgency": "high"})
        self.assertTrue(is_valid_schema(parsed))

    def test_schema_rejects_extra_keys_and_unknown_labels(self) -> None:
        self.assertFalse(
            is_valid_schema({"intent": "refund", "urgency": "high", "reason": "x"})
        )
        self.assertFalse(is_valid_schema({"intent": "other", "urgency": "high"}))
        self.assertFalse(is_valid_schema(None))

    def test_compute_metrics(self) -> None:
        rows = [
            {
                "reference": {"intent": "refund", "urgency": "high"},
                "parsed": {"intent": "refund", "urgency": "high"},
            },
            {
                "reference": {"intent": "account", "urgency": "medium"},
                "parsed": {"intent": "account", "urgency": "low"},
            },
            {
                "reference": {"intent": "product", "urgency": "low"},
                "parsed": None,
            },
        ]
        self.assertEqual(
            compute_metrics(rows),
            {
                "examples": 3,
                "json_valid_rate": 0.6667,
                "schema_valid_rate": 0.6667,
                "intent_accuracy": 0.6667,
                "urgency_accuracy": 0.3333,
                "joint_accuracy": 0.3333,
            },
        )


if __name__ == "__main__":
    unittest.main()
