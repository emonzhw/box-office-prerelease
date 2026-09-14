n = 3222 films

| feature_set   | model             |   n_features |   in_sample_adj_r2 |   cv_r2_mean |   cv_r2_std |   temporal_r2_train_pre2010 |
|:--------------|:------------------|-------------:|-------------------:|-------------:|------------:|----------------------------:|
| replication   | ols               |            6 |              0.537 |        0.528 |       0.052 |                       0.283 |
| full          | ols               |           28 |              0.665 |        0.657 |       0.044 |                       0.653 |
| full          | gradient_boosting |           28 |            nan     |        0.611 |       0.036 |                       0.598 |
| pre_release   | ols               |           25 |              0.469 |        0.458 |       0.045 |                       0.434 |
| pre_release   | gradient_boosting |           25 |            nan     |        0.388 |       0.045 |                       0.372 |
| budget_only   | ols               |            1 |              0.434 |        0.428 |       0.047 |                       0.46  |
