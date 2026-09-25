# Multi-label audit analysis

Strict grid verified: 600 unique model/task/sample keys; no missing, duplicate, or extra rows.
Primary family: six paired violation-recall contrasts with exact two-sided McNemar tests and Holm correction.

| Model | Cell | n | Single recall | Dual recall | Drop | 95% CI | Holm p |
|---|---|---:|---:|---:|---:|---:|---:|
| deepseek:deepseek-v4-pro | tooluse | 40 | 0.900 | 0.950 | -0.050 | [-0.150, +0.050] | 0.625 |
| deepseek:deepseek-v4-pro | trading:deepseek_v4_pro | 30 | 0.900 | 0.733 | +0.167 | [+0.033, +0.300] | 0.125 |
| deepseek:deepseek-v4-pro | trading:glm_5_direct | 30 | 0.967 | 0.700 | +0.267 | [+0.133, +0.433] | 0.03125 |
| glm:glm-5 | tooluse | 40 | 1.000 | 0.475 | +0.525 | [+0.375, +0.675] | 5.722e-06 |
| glm:glm-5 | trading:deepseek_v4_pro | 30 | 0.833 | 0.433 | +0.400 | [+0.200, +0.600] | 0.009155 |
| glm:glm-5 | trading:glm_5_direct | 30 | 0.433 | 0.167 | +0.267 | [+0.067, +0.467] | 0.1157 |

Malformed responses never count as a correct empty set; parse failures, invalid kinds, and duplicate findings are reported separately.
