"""
Unit tests for the DataCache module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache import DataCache


@pytest.fixture
def tmp_cache(tmp_path):
    return DataCache(cache_dir=tmp_path)


def make_test_series(n: int = 24) -> pd.Series:
    dates = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.Series(np.random.randn(n), index=dates, name="test_series")


class TestDataCache:
    def test_save_and_load(self, tmp_cache):
        series = make_test_series()
        tmp_cache.save("test_key", series)
        loaded = tmp_cache.load("test_key")
        pd.testing.assert_series_equal(series, loaded, check_names=False)

    def test_exists_after_save(self, tmp_cache):
        assert not tmp_cache.exists("my_key")
        tmp_cache.save("my_key", make_test_series())
        assert tmp_cache.exists("my_key")

    def test_load_missing_raises(self, tmp_cache):
        with pytest.raises(FileNotFoundError):
            tmp_cache.load("nonexistent_key")

    def test_invalidate(self, tmp_cache):
        tmp_cache.save("del_key", make_test_series())
        assert tmp_cache.exists("del_key")
        tmp_cache.invalidate("del_key")
        assert not tmp_cache.exists("del_key")

    def test_list_keys(self, tmp_cache):
        tmp_cache.save("alpha", make_test_series())
        tmp_cache.save("beta", make_test_series())
        keys = tmp_cache.list_keys()
        assert "alpha" in keys
        assert "beta" in keys

    def test_special_chars_in_key(self, tmp_cache):
        """Keys with slashes/colons should be sanitized."""
        series = make_test_series()
        tmp_cache.save("BAMLH0A0HYM2", series)
        loaded = tmp_cache.load("BAMLH0A0HYM2")
        pd.testing.assert_series_equal(series, loaded, check_names=False)

    def test_creates_dir(self, tmp_path):
        new_dir = tmp_path / "nested" / "cache"
        cache = DataCache(cache_dir=new_dir)
        assert new_dir.exists()

    def test_roundtrip_preserves_datetimeindex(self, tmp_cache):
        series = make_test_series()
        tmp_cache.save("dt_key", series)
        loaded = tmp_cache.load("dt_key")
        assert pd.api.types.is_datetime64_any_dtype(loaded.index)
        assert (loaded.index == series.index).all()
