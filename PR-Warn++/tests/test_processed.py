import tempfile
import unittest
from pathlib import Path

import numpy as np

from prwarn.data.processed import load_processed_split, origin_wind_from_degrees


class ProcessedSplitTests(unittest.TestCase):
    def test_archive_contract_and_direction_semantics(self):
        samples, nodes, history, features, horizon = 3, 2, 4, 3, 2
        x = np.zeros((samples, nodes, history, features), dtype=np.float32)
        x[:, :, -1, 0] = 350.0  # relative Wdir
        x[:, :, -1, 1] = 20.0  # Ndir
        arrays = {
            "x": x,
            "mask": np.ones_like(x),
            "delta_t": np.zeros_like(x),
            "y": np.ones((samples, nodes, horizon), dtype=np.float32),
            "y_mask": np.ones((samples, nodes, horizon), dtype=np.float32),
            "p_pc": np.ones((samples, nodes, horizon), dtype=np.float32),
            "current_y": np.ones((samples, nodes), dtype=np.float32),
            "current_y_mask": np.ones((samples, nodes), dtype=np.float32),
            "wind_from": np.full((samples, nodes), 10.0, dtype=np.float32),
            "origin_time": np.arange(samples).astype("datetime64[ns]"),
            "turbines": np.arange(nodes),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            np.savez_compressed(path, **arrays)
            split = load_processed_split(path)
        self.assertEqual(len(split), samples)
        direction = origin_wind_from_degrees(
            split.x,
            {"Wdir": 0, "Ndir": 1},
            wind_direction_feature="Wdir",
            mode="relative_plus_nacelle",
            nacelle_direction_feature="Ndir",
        )
        np.testing.assert_array_equal(direction, np.full((samples, nodes), 10.0))

    def test_nonbinary_mask_is_rejected(self):
        samples, nodes, history, features, horizon = 1, 1, 1, 1, 1
        arrays = {
            "x": np.zeros((samples, nodes, history, features)),
            "mask": np.full((samples, nodes, history, features), 0.5),
            "delta_t": np.zeros((samples, nodes, history, features)),
            "y": np.zeros((samples, nodes, horizon)),
            "y_mask": np.ones((samples, nodes, horizon)),
            "p_pc": np.zeros((samples, nodes, horizon)),
            "current_y": np.zeros((samples, nodes)),
            "current_y_mask": np.ones((samples, nodes)),
            "wind_from": np.zeros((samples, nodes)),
            "origin_time": np.arange(samples).astype("datetime64[ns]"),
            "turbines": np.arange(nodes),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                load_processed_split(path)


if __name__ == "__main__":
    unittest.main()
