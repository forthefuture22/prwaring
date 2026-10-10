import unittest

from prwarn.experiments.matrix import (
    build_experiment_matrix,
    experiment_specs,
    make_experiment_id,
)


class ExperimentMatrixTests(unittest.TestCase):
    def test_contains_documented_ablation_and_probability_ids(self):
        logical_ids = {spec.logical_id for spec in experiment_specs()}
        self.assertTrue({f"A{i}" for i in range(12)}.issubset(logical_ids))
        self.assertTrue({f"G{i}" for i in range(1, 7)}.issubset(logical_ids))
        self.assertTrue({f"B{i}" for i in range(10)}.issubset(logical_ids))

    def test_expansion_has_unique_stable_ids_and_no_results_claim(self):
        first = build_experiment_matrix(seeds=[11, 12])
        second = build_experiment_matrix(seeds=[11, 12])
        self.assertEqual(first, second)
        identifiers = [job["experiment_id"] for job in first]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertTrue(all(job["evidence_status"] == "planned_not_run" for job in first))

    def test_hash_changes_with_configuration(self):
        first = make_experiment_id("A0", "main", 1, {"x": 1})
        same = make_experiment_id("A0", "main", 1, {"x": 1})
        changed = make_experiment_id("A0", "main", 1, {"x": 2})
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)

    def test_group_filter(self):
        jobs = build_experiment_matrix(seeds=[1], groups=["joint_probability"])
        self.assertEqual({job["logical_id"] for job in jobs}, {f"G{i}" for i in range(1, 7)})


if __name__ == "__main__":
    unittest.main()
