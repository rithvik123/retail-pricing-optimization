# Demand Model Diagnostics

Diagnostics are calculated on the time-based test split using the retail modeling-ready feature table.

## Overall Test Metrics

- MAE: 0.1972
- RMSE: 0.5593
- SMAPE: 0.1120
- WAPE: 0.1388

## Quantity Bin Metrics

| Actual Quantity Bin | Rows | Actual Units | Predicted Units | MAE | WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11+ | 628 | 9,195 | 6,430 | 4.9325 | 0.3369 |
| 6-10 | 3,999 | 28,244 | 23,194 | 1.6073 | 0.2276 |
| 3-5 | 25,639 | 90,544 | 79,546 | 0.6722 | 0.1903 |
| 2 | 65,864 | 131,728 | 122,897 | 0.3149 | 0.1575 |
| 1 | 292,911 | 292,911 | 318,533 | 0.0997 | 0.0997 |

## Highest-WAPE Departments

| Department | Rows | WAPE | MAE |
| --- | ---: | ---: | ---: |
| FLORAL | 628 | 0.1774 | 0.2096 |
| SALAD BAR | 1,156 | 0.1731 | 0.2454 |
| PASTRY | 5,792 | 0.1720 | 0.2360 |
| MEAT | 12,671 | 0.1631 | 0.2391 |
| DRUG GM | 44,803 | 0.1611 | 0.2156 |
| PRODUCE | 31,607 | 0.1545 | 0.2379 |
| NUTRITION | 5,522 | 0.1374 | 0.1907 |
| GROCERY | 255,998 | 0.1354 | 0.1945 |
| SEAFOOD | 659 | 0.1073 | 0.1246 |
| MEAT-PCKGD | 16,842 | 0.1047 | 0.1451 |

## Highest-WAPE Categories

| Category | Rows | WAPE | MAE |
| --- | ---: | ---: | ---: |
| PWDR/CRYSTL DRNK MX | 1,869 | 0.4189 | 0.9607 |
| PEPPERS-ALL | 1,311 | 0.2858 | 0.4702 |
| CIGARETTES | 1,498 | 0.2833 | 0.4645 |
| SOUP | 8,925 | 0.2828 | 0.5256 |
| CITRUS | 2,138 | 0.2782 | 0.7400 |
| CHRISTMAS  SEASONAL | 1,202 | 0.2770 | 0.4439 |
| ISOTONIC DRINKS | 1,404 | 0.2709 | 0.4323 |
| BABY FOODS | 3,039 | 0.2696 | 0.4247 |
| CAT FOOD | 2,674 | 0.2620 | 0.4555 |
| BREAKFAST SWEETS | 1,221 | 0.2612 | 0.4862 |

## Interpretation

- The curated model is now modeling ordinary retail unit demand rather than fuel/coupon pseudo-units.
- Remaining error should be reviewed by high-WAPE departments/categories before using recommendations in production.
- The next model improvement should focus on better treatment of sparse product-store histories and promotion timing.