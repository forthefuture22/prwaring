import unittest

from prwarn.experiments.resolve import (
    apply_dotted_overrides,
    command_template,
    execution_stage,
    post_command_templates,
)


class ExperimentResolveTests(unittest.TestCase):
    def test_dotted_override_does_not_mutate_base(self):
        base = {"model": {"x": 1}, "flow": {"steps": 8}}
        resolved = apply_dotted_overrides(base, {"model.x": 2, "new.value": [1]})
        self.assertEqual(base["model"]["x"], 1)
        self.assertEqual(resolved["model"]["x"], 2)
        self.assertEqual(resolved["new"]["value"], [1])

    def test_stage_mapping(self):
        self.assertEqual(execution_stage({"logical_id": "A9"}), "deterministic")
        self.assertEqual(execution_stage({"logical_id": "G5"}), "neural_baseline")
        self.assertEqual(execution_stage({"logical_id": "B6"}), "marginal_baseline")

    def test_new_code_paths_are_materialized_not_manual_gates(self):
        jobs = [
            {"logical_id": "A10", "variant": "risk_core", "experiment_id": "x", "overrides": {}},
            {"logical_id": "A11", "variant": "oracle_era5", "experiment_id": "y", "overrides": {"data.weather_protocol": "oracle_era5"}},
            {"logical_id": "B6", "variant": "marginal", "experiment_id": "z", "overrides": {"marginal.method": "quantile"}},
        ]
        for job in jobs:
            self.assertNotEqual(command_template(job, "config.json")[0], "MANUAL_GATE")
        self.assertTrue(post_command_templates(jobs[-1], "config.json"))


if __name__ == "__main__":
    unittest.main()
