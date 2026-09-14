import pandas as pd

# Measured only after a film opens (or, for popularity, at the 2017 scrape) — unusable for forecasting.
POST_RELEASE = ["log_vote_count", "vote_average", "log_popularity"]

PRE_RELEASE_NUMERIC = ["log_budget", "runtime", "release_year", "gdp_real_prior_year", "is_english"]

# The COGS 108 team model: same-year nominal GDP and raw (unlogged) engagement counts.
REPLICATION = ["log_budget", "runtime", "vote_average", "vote_count", "popularity", "gdp_same_year"]

MIN_GENRE_COUNT = 30
SEASONS = {"summer": [5, 6, 7], "holiday": [11, 12]}


def genre_dummies(movies, min_count=MIN_GENRE_COUNT):
    exploded = movies["genre_list"].explode().dropna()
    keep = exploded.value_counts()
    keep = keep[keep >= min_count].index
    return pd.DataFrame(
        {f"genre_{g.lower().replace(' ', '_')}": movies["genre_list"].apply(lambda gl, g=g: int(g in gl)) for g in sorted(keep)},
        index=movies.index,
    )


def season_dummies(movies):
    return pd.DataFrame(
        {f"season_{name}": movies["release_month"].isin(months).astype(int) for name, months in SEASONS.items()},
        index=movies.index,
    )


def build_matrix(movies, feature_set):
    if feature_set == "replication":
        return movies[REPLICATION].copy()
    if feature_set == "budget_only":
        return movies[["log_budget"]].copy()

    pre = pd.concat([movies[PRE_RELEASE_NUMERIC], genre_dummies(movies), season_dummies(movies)], axis=1)
    if feature_set == "pre_release":
        return pre
    if feature_set == "full":
        return pd.concat([pre, movies[POST_RELEASE]], axis=1)
    raise ValueError(f"unknown feature set: {feature_set!r}")
