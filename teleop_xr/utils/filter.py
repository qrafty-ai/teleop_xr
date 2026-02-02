from typing import List, Optional, Union
import numpy as np


class WeightedMovingFilter:
    def __init__(self, weights: Union[List[float], np.ndarray], data_size: int = 14):
        self._window_size: int = len(weights)
        self._weights: np.ndarray = np.array(weights)
        assert np.isclose(np.sum(self._weights), 1.0), (
            "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        )
        self._data_size: int = data_size
        self._filtered_data: Optional[np.ndarray] = None
        self._data_queue: List[np.ndarray] = []

    def _apply_filter(self) -> np.ndarray:
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        temp_filtered_data = np.zeros(self._data_size)
        for i in range(self._data_size):
            temp_filtered_data[i] = np.convolve(
                data_array[:, i], self._weights, mode="valid"
            )[-1]

        return temp_filtered_data

    def reset(self) -> None:
        self._data_queue = []
        self._filtered_data = None

    def add_data(self, new_data: np.ndarray) -> None:
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return  # skip duplicate data

        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    def data_ready(self) -> bool:
        return self._filtered_data is not None

    @property
    def filtered_data(self) -> np.ndarray:
        if self._filtered_data is None:
            raise ValueError("Filtered data is not available yet")
        return self._filtered_data
