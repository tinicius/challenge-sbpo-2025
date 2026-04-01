import csv
import json
import os
import tempfile
import unittest

from runner.consolidate import consolidate_results


class ConsolidateSimilarityTest(unittest.TestCase):
    def test_generates_similarity_and_distribution_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "runs_a_20260401_000000.jsonl")
            output_dir = os.path.join(tmpdir, "out")

            records = [
                {
                    "algorithm": "simple",
                    "instance": "instance_001.txt",
                    "instance_path": "datasets/a/instance_001.txt",
                    "run_id": 0,
                    "seed": 42,
                    "objective": 5.0,
                    "time_s": 0.1,
                    "feasible": True,
                    "timed_out": False,
                    "total_items": 10,
                    "num_aisles": 2,
                    "num_orders": 4,
                    "selected_orders": [0, 1, 2, 4],
                    "visited_aisles": [1, 3],
                    "selected_items": [0, 1, 2, 3, 4],
                    "params": {},
                },
                {
                    "algorithm": "simple_multi",
                    "instance": "instance_001.txt",
                    "instance_path": "datasets/a/instance_001.txt",
                    "run_id": 0,
                    "seed": 42,
                    "objective": 4.0,
                    "time_s": 0.2,
                    "feasible": True,
                    "timed_out": False,
                    "total_items": 12,
                    "num_aisles": 3,
                    "num_orders": 3,
                    "selected_orders": [0, 2, 3],
                    "visited_aisles": [1, 3, 4],
                    "selected_items": [0, 1, 2, 3, 4],
                    "params": {},
                },
            ]

            with open(jsonl_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            summary_path = consolidate_results(jsonl_path, output_dir)

            self.assertTrue(os.path.exists(summary_path))

            suffix = "a_20260401_000000"
            similarity_path = os.path.join(output_dir, f"similarity_{suffix}.csv")
            distribution_path = os.path.join(output_dir, f"distribution_{suffix}.csv")

            self.assertTrue(os.path.exists(similarity_path))
            self.assertTrue(os.path.exists(distribution_path))

            with open(similarity_path, "r") as f:
                similarity_rows = list(csv.DictReader(f))
            self.assertEqual(len(similarity_rows), 1)

            row = similarity_rows[0]
            self.assertEqual(row["instance"], "instance_001.txt")
            self.assertEqual(row["algorithm_left"], "simple")
            self.assertEqual(row["algorithm_right"], "simple_multi")
            self.assertAlmostEqual(float(row["orders_jaccard"]), 0.4)
            self.assertAlmostEqual(float(row["aisles_jaccard"]), 2 / 3)
            self.assertAlmostEqual(float(row["items_jaccard"]), 1.0)

            with open(distribution_path, "r") as f:
                distribution_rows = list(csv.DictReader(f))
            self.assertEqual(len(distribution_rows), 2)
            self.assertEqual(
                {r["algorithm"] for r in distribution_rows},
                {"simple", "simple_multi"},
            )


if __name__ == "__main__":
    unittest.main()
