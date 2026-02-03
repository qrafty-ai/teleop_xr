import pytest
import numpy as np
from teleop_xr.utils.filter import WeightedMovingFilter


def test_weighted_moving_filter_init():
    f = WeightedMovingFilter(weights=[0.5, 0.5], data_size=3)
    assert f._window_size == 2
    assert f._data_size == 3
    assert f._filtered_data is None

    with pytest.raises(AssertionError):
        WeightedMovingFilter(weights=[0.5, 0.4])


def test_weighted_moving_filter_add_data():
    f = WeightedMovingFilter(weights=[0.5, 0.5], data_size=2)

    d1 = np.array([1.0, 2.0])
    f.add_data(d1)
    assert f.data_ready() is True
    np.testing.assert_array_equal(f.filtered_data, d1)

    d2 = np.array([3.0, 4.0])
    f.add_data(d2)

    expected = np.array([2.0, 3.0])
    np.testing.assert_array_equal(f.filtered_data, expected)


def test_weighted_moving_filter_reset():
    f = WeightedMovingFilter(weights=[1.0], data_size=2)
    f.add_data(np.array([1.0, 2.0]))
    assert f.data_ready() is True

    f.reset()
    assert f.data_ready() is False
    with pytest.raises(ValueError):
        _ = f.filtered_data


def test_weighted_moving_filter_duplicate():
    f = WeightedMovingFilter(weights=[0.5, 0.5], data_size=2)
    d1 = np.array([1.0, 2.0])
    f.add_data(d1)
    f.add_data(d1)

    assert len(f._data_queue) == 1


def test_weighted_moving_filter_window_size():
    f = WeightedMovingFilter(weights=[0.5, 0.5], data_size=1)

    f.add_data(np.array([1.0]))
    f.add_data(np.array([2.0]))
    f.add_data(np.array([3.0]))

    assert len(f._data_queue) == 2
    np.testing.assert_array_equal(f._data_queue[0], np.array([2.0]))
    np.testing.assert_array_equal(f._data_queue[1], np.array([3.0]))
