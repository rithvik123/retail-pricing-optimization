from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    column: str
    check: str
    failing_rows: int


def validate_transactions(transactions: pd.DataFrame) -> list[ValidationIssue]:
    checks: list[ValidationIssue] = []

    if "quantity" in transactions:
        failing = int((transactions["quantity"] <= 0).sum())
        if failing:
            checks.append(ValidationIssue("quantity", "must be positive", failing))

    if "sales_value" in transactions:
        failing = int((transactions["sales_value"] < 0).sum())
        if failing:
            checks.append(ValidationIssue("sales_value", "must be non-negative", failing))

    if "effective_unit_price" in transactions:
        failing = int((transactions["effective_unit_price"] <= 0).sum())
        if failing:
            checks.append(ValidationIssue("effective_unit_price", "must be positive", failing))

    if "discount_percentage" in transactions:
        invalid = transactions["discount_percentage"].notna() & (
            (transactions["discount_percentage"] < 0)
            | (transactions["discount_percentage"] > 1.5)
        )
        failing = int(invalid.sum())
        if failing:
            checks.append(
                ValidationIssue("discount_percentage", "must be between 0 and 150%", failing)
            )

    if "week_no" in transactions:
        invalid = transactions["week_no"].notna() & (
            (transactions["week_no"] < 1) | (transactions["week_no"] > 102)
        )
        failing = int(invalid.sum())
        if failing:
            checks.append(ValidationIssue("week_no", "must be between 1 and 102", failing))

    for column in ("product_id", "store_id"):
        if column in transactions:
            failing = int(transactions[column].isna().sum())
            if failing:
                checks.append(ValidationIssue(column, "must not be null", failing))

    return checks


def validation_issues_to_frame(issues: list[ValidationIssue]) -> pd.DataFrame:
    return pd.DataFrame([issue.__dict__ for issue in issues])

