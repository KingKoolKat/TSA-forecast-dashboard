## 📈 Hyperparameter Tuning Summary

**Next-week MAPE improved from 6.52% → 3.84%** (**41.1%** relative improvement) by tuning trend/seasonality/holiday priors.

**Best config:**
- `changepoint_prior_scale`: **0.5**
- `seasonality_mode`: **additive**
- `seasonality_prior_scale`: **1**
- `holidays_prior_scale`: **10**
- `changepoint_range`: **0.8**

**Charts:**
![Best vs Default](reports/default_vs_best.png)

![MAPE vs CPS by mode (faceted by changepoint_range)](reports/mape_vs_cps_by_mode_facet_crange.png)

<details><summary>More comparisons</summary>

![Box by cps](reports/box_by_cps.png)
![Box by mode](reports/box_by_mode.png)
![Box by seasonality_prior_scale](reports/box_by_sps.png)
![Box by holidays_prior_scale](reports/box_by_hps.png)
![Box by changepoint_range](reports/box_by_crange.png)
![Heatmap SPS×HPS @ best cps & crange](reports/heatmap_sps_hps_at_best_cps_crange.png)

</details>