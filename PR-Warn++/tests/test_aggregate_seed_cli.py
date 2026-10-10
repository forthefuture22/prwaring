import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from prwarn.cli.aggregate_seed_runs import main


class AggregateSeedCliTests(unittest.TestCase):
    def test_writes_common_nested_metric_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps({"test": {"crps": 2.0}}), encoding="utf-8")
            second.write_text(json.dumps({"test": {"crps": 4.0}}), encoding="utf-8")
            output = root / "summary.json"
            argv = [
                "aggregate_seed_runs",
                "--metrics",
                str(first),
                str(second),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["run_count"], 2)
            self.assertEqual(result["summary"]["test.crps"]["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
