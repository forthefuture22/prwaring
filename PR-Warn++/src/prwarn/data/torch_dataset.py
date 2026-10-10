"""PyTorch adapters for processed PR-Warn++ split archives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from prwarn.graphs.builders import directional_graph
from prwarn.models.backbone import mix_graphs_from_weights

from .processed import ProcessedSplit


class WindowTensorDataset(Dataset):
    def __init__(self, split: ProcessedSplit, wind_from: np.ndarray) -> None:
        if wind_from.shape != split.x.shape[:2]:
            raise ValueError("wind_from must have [sample,node]")
        self.split = split
        self.wind_from = wind_from.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.int64]:
        item = {
            "x": self.split.x[index],
            "mask": self.split.mask[index],
            "delta_t": self.split.delta_t[index],
            "y": self.split.y[index],
            "y_mask": self.split.y_mask[index],
            "p_pc": self.split.p_pc[index],
            "current_y": self.split.current_y[index],
            "current_y_mask": self.split.current_y_mask[index],
            "origin_time": self.split.origin_time[index].astype("datetime64[ns]").astype(np.int64),
            "wind_from": self.wind_from[index],
        }
        if self.split.future_weather is not None:
            item["future_weather"] = self.split.future_weather[index]
            item["future_weather_mask"] = self.split.future_weather_mask[index]
        return item


@dataclass
class DirectionalGraphCollator:
    coordinates: np.ndarray
    distance_scale: float
    sigma_degrees: float
    sector_degrees: float

    def __call__(self, examples: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        batch = default_collate(examples)
        wind = batch["wind_from"].numpy()
        adjacency = directional_graph(
            self.coordinates,
            wind,
            distance_scale=self.distance_scale,
            sigma_degrees=self.sigma_degrees,
            sector_degrees=self.sector_degrees,
        )
        batch["directional_graph"] = torch.from_numpy(adjacency)
        return batch


class DeterministicCacheDataset(Dataset):
    """Lazy HDF5 reader for flow training or scenario export."""

    def __init__(
        self,
        path: str,
        *,
        error_mean: np.ndarray,
        error_scale: np.ndarray,
        mode: str = "train",
    ) -> None:
        if mode not in {"train", "export"}:
            raise ValueError("mode must be train or export")
        self.path = str(path)
        self.error_mean = np.asarray(error_mean, dtype=np.float32)
        self.error_scale = np.asarray(error_scale, dtype=np.float32)
        self.mode = mode
        self._file = None
        with h5py.File(self.path, "r") as source:
            self.length = len(source["y"])

    def __len__(self) -> int:
        return self.length

    def _source(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.int64]:
        source = self._source()
        error = source["y"][index] - source["y_det"][index]
        item = {
            "normalized_error": ((error - self.error_mean) / self.error_scale).astype(np.float32),
            "condition": source["hidden"][index],
            "y_mask": source["y_mask"][index],
            "graph_weights": source["graph_weights"][index],
            "wind_from": source["wind_from"][index],
        }
        if self.mode == "export":
            item.update(
                {
                    "y": source["y"][index],
                    "y_det": source["y_det"][index],
                    "current_y": source["current_y"][index],
                    "current_y_mask": source["current_y_mask"][index],
                    "origin_time": np.int64(source["origin_time"][index]),
                }
            )
        return item


@dataclass
class FlowGraphCollator:
    coordinates: np.ndarray
    static_graphs: np.ndarray
    adaptive_graph: np.ndarray
    distance_scale: float
    sigma_degrees: float
    sector_degrees: float

    def __call__(self, examples: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        batch = default_collate(examples)
        directional = directional_graph(
            self.coordinates,
            batch["wind_from"].numpy(),
            distance_scale=self.distance_scale,
            sigma_degrees=self.sigma_degrees,
            sector_degrees=self.sector_degrees,
        )
        batch["adjacency"] = mix_graphs_from_weights(
            torch.from_numpy(self.static_graphs),
            torch.from_numpy(directional),
            torch.from_numpy(self.adaptive_graph),
            batch["graph_weights"],
        )
        return batch
