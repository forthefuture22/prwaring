import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from prwarn.cli.preprocess_sdwpf import main
from prwarn.data.processed import load_processed_split


class PreprocessCliTests(unittest.TestCase):
    def test_end_to_end_history_only_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for step in range(80):
                for turbine in (1, 2):
                    speed = 4.0 + 0.1 * (step % 20)
                    rows.append(
                        {
                            "TurbID": turbine,
                            "Tmstamp": pd.Timestamp("2021-01-01") + pd.Timedelta(minutes=10 * step),
                            "Patv": min(2000.0, 8.0 * speed**3 + turbine),
                            "Wspd": speed,
                            "Wdir": -10.0,
                            "Ndir": 20.0,
                            "Sp": 101325.0,
                            "T2m": 288.15,
                        }
                    )
            raw_path = root / "raw.csv"
            location_path = root / "locations.csv"
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            pd.DataFrame({"TurbID": [1, 2], "x": [0.0, 500.0], "y": [0.0, 0.0]}).to_csv(
                location_path, index=False
            )
            output = root / "processed"
            argv = [
                "preprocess_sdwpf",
                "--input", str(raw_path),
                "--locations", str(location_path),
                "--output-dir", str(output),
                "--rated-power", "2000",
                "--history", "4",
                "--horizon", "2",
                "--curve-bins", "5",
                "--curve-min-count", "2",
                "--features", "Wspd", "Wdir", "Ndir", "Patv", "Sp", "T2m",
            ]
            with patch.object(sys, "argv", argv):
                main()

            train = load_processed_split(output / "train.npz")
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["weather_protocol"], "history_only")
            self.assertEqual(metadata["feature_scaler"]["fit_split"], "train")
            np.testing.assert_allclose(train.wind_from, 10.0)
            # History-only P_pc persists the origin value across every horizon.
            np.testing.assert_allclose(train.p_pc[..., 0], train.p_pc[..., 1])
            self.assertTrue(np.isfinite(train.x).all())
            self.assertTrue((output / "a_geo.npy").exists())
            self.assertTrue((output / "a_corr.npy").exists())


if __name__ == "__main__":
    unittest.main()
