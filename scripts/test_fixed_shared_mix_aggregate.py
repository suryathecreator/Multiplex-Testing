import unittest

from aggregate_aime_train100_fixed_shared_mix4096 import (
    MIXED_CONDITION,
    check_mixed_sanity,
    compute_repeat_metrics,
    merge_prompt_rows,
)


def prompt_row(prompt_index, max_k, true_ks, base_cost):
    row = {
        "prompt_index": prompt_index,
        "usable_sample_count": max_k,
    }
    for k in (1, 2, 4, 8, 16, 32):
        if k <= max_k:
            row[f"pass_at_{k}"] = k in true_ks
            row[f"cost_at_{k}"] = base_cost * k
    return row


class FixedSharedMixAggregateTests(unittest.TestCase):
    def test_mixed_half_fixed_half_shared_scoring(self):
        fixed = {0: prompt_row(0, 4, true_ks={2, 4}, base_cost=10)}
        shared = {0: prompt_row(0, 4, true_ks={1, 2, 4}, base_cost=20)}
        _, mixed_prompt, _, per_repeat_mixed = compute_repeat_metrics(
            repeat_index=0,
            seed="409600",
            fixed_rows=fixed,
            shared_rows=shared,
            max_k=4,
        )

        by_k = {int(row["k"]): row for row in mixed_prompt}
        self.assertEqual(sorted(by_k), [2, 4])
        self.assertEqual(by_k[2]["component_k"], 1)
        self.assertTrue(by_k[2]["correct"])
        self.assertEqual(by_k[2]["cost_tokens"], 30)
        self.assertEqual(by_k[4]["component_k"], 2)
        self.assertTrue(by_k[4]["correct"])
        self.assertEqual(by_k[4]["cost_tokens"], 60)

        repeat_by_k = {int(row["k"]): row for row in per_repeat_mixed}
        self.assertEqual(repeat_by_k[4]["fixed_component_accuracy"], 1.0)
        self.assertEqual(repeat_by_k[4]["shared_component_accuracy"], 1.0)

    def test_mixed_pass_one_is_omitted(self):
        fixed = {0: prompt_row(0, 4, true_ks=set(), base_cost=10)}
        shared = {0: prompt_row(0, 4, true_ks=set(), base_cost=20)}
        _, mixed_prompt, _, _ = compute_repeat_metrics(
            repeat_index=0,
            seed="409600",
            fixed_rows=fixed,
            shared_rows=shared,
            max_k=4,
        )
        self.assertNotIn(1, {int(row["k"]) for row in mixed_prompt})

    def test_duplicate_prompt_shards_raise(self):
        merged = {}
        merge_prompt_rows(merged, [{"prompt_index": 7}], source="shard-a")
        with self.assertRaises(ValueError):
            merge_prompt_rows(merged, [{"prompt_index": 7}], source="shard-b")

    def test_sanity_check_passes_for_valid_or_rows(self):
        mixed_rows = [
            {
                "repeat_index": 0,
                "seed": "409600",
                "prompt_index": 0,
                "condition": MIXED_CONDITION,
                "k": 2,
                "component_k": 1,
                "correct": True,
                "fixed_component_correct": False,
                "shared_component_correct": True,
            }
        ]
        repeat_rows = [
            {
                "repeat_index": 0,
                "seed": "409600",
                "condition": MIXED_CONDITION,
                "k": 2,
                "component_k": 1,
                "accuracy": 1.0,
                "fixed_component_accuracy": 0.0,
                "shared_component_accuracy": 1.0,
            }
        ]
        checks, violations = check_mixed_sanity(mixed_rows, repeat_rows)
        self.assertGreaterEqual(len(checks), 2)
        self.assertEqual(violations, [])

    def test_sanity_check_fails_for_invalid_or_rows(self):
        mixed_rows = [
            {
                "repeat_index": 0,
                "seed": "409600",
                "prompt_index": 0,
                "condition": MIXED_CONDITION,
                "k": 2,
                "component_k": 1,
                "correct": False,
                "fixed_component_correct": True,
                "shared_component_correct": False,
            }
        ]
        repeat_rows = [
            {
                "repeat_index": 0,
                "seed": "409600",
                "condition": MIXED_CONDITION,
                "k": 2,
                "component_k": 1,
                "accuracy": 0.0,
                "fixed_component_accuracy": 1.0,
                "shared_component_accuracy": 0.0,
            }
        ]
        _, violations = check_mixed_sanity(mixed_rows, repeat_rows)
        self.assertEqual(len(violations), 2)


if __name__ == "__main__":
    unittest.main()
