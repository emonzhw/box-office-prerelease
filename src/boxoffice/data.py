import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_RUNTIME = 60


def parse_genres(raw):
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return [g["name"] for g in json.loads(raw)]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def load_tmdb(path):
    return pd.read_csv(Path(path))


def clean_movies(raw):
    df = raw.copy()
    for col in ("budget", "revenue"):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    if "status" in df.columns:
        df = df[df["status"] == "Released"]

    # Zero budget/revenue in TMDB means "unknown", and sub-60-minute entries are shorts or data errors.
    df = df.dropna(subset=["budget", "revenue", "runtime", "release_date"])
    df = df[df["runtime"] >= MIN_RUNTIME]

    df["genre_list"] = df["genres"].apply(parse_genres)
    df["release_year"] = df["release_date"].dt.year.astype(int)
    df["release_month"] = df["release_date"].dt.month.astype(int)
    df["log_revenue"] = np.log1p(df["revenue"])
    df["log_budget"] = np.log1p(df["budget"])
    df["log_vote_count"] = np.log1p(df["vote_count"])
    df["log_popularity"] = np.log1p(df["popularity"])
    df["is_english"] = (df["original_language"] == "en").astype(int)
    return df.reset_index(drop=True)


def load_gdp(path):
    gdp = pd.read_csv(Path(path))
    gdp = gdp.rename(columns={"date": "year", "level-current": "gdp_nominal", "level-chained": "gdp_real"})
    gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce")
    return gdp.dropna(subset=["year"]).astype({"year": int})[["year", "gdp_nominal", "gdp_real"]]


def attach_gdp(movies, gdp):
    same_year = gdp.rename(columns={"year": "release_year", "gdp_nominal": "gdp_same_year"})
    # A studio greenlighting a film only knows last year's GDP, so the forecasting model gets the lagged value.
    prior_year = gdp.assign(release_year=gdp["year"] + 1).rename(columns={"gdp_real": "gdp_real_prior_year"})
    out = movies.merge(same_year[["release_year", "gdp_same_year"]], on="release_year", how="left")
    out = out.merge(prior_year[["release_year", "gdp_real_prior_year"]], on="release_year", how="left")
    return out.dropna(subset=["gdp_same_year", "gdp_real_prior_year"]).reset_index(drop=True)
