# Modeling-Ready Dataset Report

The raw product-store-week feature table is preserved, but demand modeling uses a retail-focused table that removes non-merchandise rows and extreme anomalies.

- Input rows: 2,352,462
- Output rows: 2,338,832
- Excluded rows: 13,630
- Quantity cap: 100
- Input units: 207,588,134
- Output units: 3,335,715
- Input revenue: $7,817,991.99
- Output revenue: $7,284,136.02

## Exclusion Rules

| Rule | Rows |
| --- | ---: |
| non_merchandise_commodity | 11,477 |
| non_merchandise_department | 9,463 |
| unit_price_below_minimum | 5,517 |
| quantity_above_99_9_percentile | 5,419 |
| discount_above_100_percent | 772 |

Primary rationale: fuel and coupon/miscellaneous rows encode quantity in non-standard units, producing huge demand values at near-zero prices. They should not train product pricing recommendations.