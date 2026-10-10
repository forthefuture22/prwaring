import unittest

import pandas as pd

from prwarn.data.audit import audit_sdwpf_frame


class RawAuditTests(unittest.TestCase):
    def test_audit_reports_grid_gap_without_hiding_it(self):
        rows = []
        for turbine in (1, 2):
            for step in (0, 1, 3):
                rows.append(
                    {
                        "TurbID": turbine,
                        "Tmstamp": pd.Timestamp("2021-01-01") + pd.Timedelta(minutes=10 * step),
                        "Patv": 100.0,
                        "Wspd": 5.0,
                    }
                )
        report = audit_sdwpf_frame(pd.DataFrame(rows), expected_turbines=2)
        self.assertEqual(report["grid_row_gap"], 2)
        self.assertTrue(any("reindex" in issue for issue in report["issues"]))
        self.assertEqual(report["reason_counts"]["valid"], 6)

    def test_duplicate_blocks_gate(self):
        frame = pd.DataFrame(
            {
                "TurbID": [1, 1],
                "Tmstamp": ["2021-01-01", "2021-01-01"],
                "Patv": [1.0, 1.0],
                "Wspd": [1.0, 1.0],
            }
        )
        report = audit_sdwpf_frame(frame, expected_turbines=1)
        self.assertFalse(report["gate_ready"])


if __name__ == "__main__":
    unittest.main()
