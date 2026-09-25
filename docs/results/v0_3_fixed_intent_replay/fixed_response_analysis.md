# Fixed-response two-by-two replay analysis

The source gate fixes each raw provider response tape and verifies identical pre-risk decisions across replay destinations. Downstream risk approval and fills remain execution-endogenous. Response-origin contrasts are descriptive, not execution-randomized.

## Overall total-return decomposition

| Estimand | Estimate | Seed-cluster 95% CI |
|---|---:|---:|
| execution_within_I0 | -0.019335 | [-0.027295, -0.012122] |
| execution_within_I1 | -0.014666 | [-0.019680, -0.010035] |
| response_origin_within_X0 | +0.001183 | [-0.007544, +0.010136] |
| response_origin_within_X1 | +0.005853 | [-0.005312, +0.017590] |
| interaction | +0.004669 | [-0.000504, +0.010533] |
| execution_shapley | -0.017001 | [-0.022994, -0.011650] |
| response_origin_shapley | +0.003518 | [-0.006112, +0.013555] |
| observed_diagonal | -0.013483 | [-0.023849, -0.002504] |

Shared-seed bootstrap: 10 clusters, 10,000 resamples, seed 20260719, using CPython random.Random (MT19937). Each replicate makes 10 randrange(10) draws with replacement and retains every model, scenario, provider sample, response origin, and replay destination within each selected seed. CIs use Hyndman-Fan type 7 linear percentiles at 2.5% and 97.5%.

## Source-arm divergence diagnostic

All 450 pairs share the first prompt. The first raw response hash agrees in 44.9%, the first parsed response in 50.9%, the full parsed-response path in 19.8%, and the full pre-risk decision path in 19.8%. The original diagonal E0/E1 contrast therefore cannot be called execution-only.

## Sharpe ranking stability under fixed response tapes

| Scenario | Response origin | tau-b(E0,E1) | 95% CI | Exact-order p | Winner E0 -> E1 |
|---|---|---:|---:|---:|---|
| synthetic_calm_trend_c0_v0_3 | E0 | +1.000 | [+0.800, +1.000] | 0.763 | glm-5.2 -> glm-5.2 |
| synthetic_calm_trend_c0_v0_3 | E1 | +1.000 | [+0.600, +1.000] | 0.261 | glm-5.2 -> glm-5.2 |
| synthetic_high_volatility_c0_v0_3 | E0 | +1.000 | [+0.600, +1.000] | 0.537 | glm-5.2 -> glm-5.2 |
| synthetic_high_volatility_c0_v0_3 | E1 | +1.000 | [+0.600, +1.000] | 0.568 | glm-5.2 -> glm-5.2 |
| synthetic_jump_tail_c0_v0_3 | E0 | +1.000 | [+0.800, +1.000] | 0.737 | glm-5.2 -> glm-5.2 |
| synthetic_jump_tail_c0_v0_3 | E1 | +0.800 | [+0.400, +1.000] | 0.327 | glm-5.2 -> glm-5 |

Ranking caveat: deepseek-v4-pro is inactive (zero return, all-hold, zero gross target exposure) in 358/360 replay rows; inactivity must not be interpreted as robustness.
