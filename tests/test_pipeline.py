import json

import numpy as np
import pandas as pd
import pytest

from boxoffice.data import attach_gdp, clean_movies, parse_genres
from boxoffice.evaluate import cv_r2, models, temporal_r2
from boxoffice.features import POST_RELEASE, build_matrix, genre_dummies


def raw_movies(n=200, seed=0):
    rng = np.random.default_rng(seed)
    genres = ["Action", "Drama", "Comedy", "Science Fiction"]
    budget = rng.lognormal(17, 1, n)
    return pd.DataFrame(
        {
            "budget": budget,
            "revenue": budget * rng.lognormal(0.5, 0.8, n),
            "runtime": rng.normal(110, 20, n).clip(40, 200),
            "release_date": pd.to_datetime("1995-01-01") + pd.to_timedelta(rng.integers(0, 365 * 20, n), unit="D"),
            "genres": [json.dumps([{"id": 1, "name": g} for g in rng.choice(genres, 2, replace=False)]) for _ in range(n)],
            "vote_count": rng.integers(1, 5000, n),
            "vote_average": rng.uniform(3, 9, n),
            "popularity": rng.lognormal(2, 1, n),
            "original_language": rng.choice(["en", "fr"], n, p=[0.9, 0.1]),
            "status": "Released",
        }
    )


def gdp_table():
    years = np.arange(1990, 2020)
    return pd.DataFrame({"year": years, "gdp_nominal": years * 10.0, "gdp_real": years * 8.0})


@pytest.fixture
def movies():
    return attach_gdp(clean_movies(raw_movies()), gdp_table())


def test_parse_genres_handles_bad_input():
    assert parse_genres('[{"id": 1, "name": "Drama"}]') == ["Drama"]
    assert parse_genres("") == []
    assert parse_genres("not json") == []
    assert parse_genres(float("nan")) == []


def test_zero_budget_and_short_runtime_are_dropped():
    raw = raw_movies(10)
    raw.loc[0, "budget"] = 0
    raw.loc[1, "runtime"] = 45
    raw.loc[2, "status"] = "Rumored"
    cleaned = clean_movies(raw)
    assert len(cleaned) <= 7
    assert (cleaned["budget"] > 0).all()
    assert (cleaned["runtime"] >= 60).all()


def test_gdp_is_lagged_one_year(movies):
    row = movies.iloc[0]
    assert row["gdp_real_prior_year"] == (row["release_year"] - 1) * 8.0
    assert row["gdp_same_year"] == row["release_year"] * 10.0


def test_pre_release_matrix_has_no_post_release_leakage(movies):
    cols = set(build_matrix(movies, "pre_release").columns)
    assert cols.isdisjoint(POST_RELEASE)
    assert cols.isdisjoint({"vote_count", "popularity", "revenue", "log_revenue"})


def test_full_matrix_is_pre_release_plus_post_release(movies):
    pre = set(build_matrix(movies, "pre_release").columns)
    full = set(build_matrix(movies, "full").columns)
    assert full == pre | set(POST_RELEASE)


def test_rare_genres_are_dropped(movies):
    assert genre_dummies(movies, min_count=10_000).empty


def test_budget_only_baseline(movies):
    assert list(build_matrix(movies, "budget_only").columns) == ["log_budget"]


def test_unknown_feature_set_raises(movies):
    with pytest.raises(ValueError):
        build_matrix(movies, "everything")


def test_models_learn_budget_signal(movies):
    X = build_matrix(movies, "pre_release").to_numpy()
    y = movies["log_revenue"].to_numpy()
    mean, _ = cv_r2(models()["ols"], X, y)
    assert mean > 0.2


def test_temporal_r2_is_nan_without_a_test_period(movies):
    X = build_matrix(movies, "pre_release").to_numpy()
    y = movies["log_revenue"].to_numpy()
    assert np.isnan(temporal_r2(models()["ols"], X, y, movies["release_year"], cutoff=3000))
