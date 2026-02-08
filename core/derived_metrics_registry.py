"""
Derived Metrics Registry.

Defines rules and formulas for creating "virtual" metrics from existing data.
Used by MetricResolver to synthesize business concepts when raw columns are missing.
"""
import polars as pl
from typing import List, Callable, Dict, Any, Optional

class DerivedMetricDefinition:
    """Definition of a derived metric rule."""
    def __init__(
        self,
        name: str,
        description: str,
        required_roles: List[str],
        required_keywords: List[List[str]],
        formula: Callable[[pl.DataFrame, List[str]], pl.Series]
    ):
        self.name = name
        self.description = description
        self.required_roles = required_roles
        # List of keyword lists. One column must match each list of keywords.
        # e.g. [["revenue", "sales"], ["spend", "cost"]] matches (revenue, ad_spend)
        self.required_keywords = required_keywords
        self.formula = formula

def _safe_div(numerator: pl.Series, denominator: pl.Series) -> pl.Series:
    """Safe division handling zeros."""
    return numerator / denominator.replace(0, None)

# --- Formula Implementations ---

def calc_roas(df: pl.DataFrame, cols: List[str]) -> pl.Series:
    """Revenue / Ad Spend"""
    return _safe_div(df[cols[0]], df[cols[1]])

def calc_cpa(df: pl.DataFrame, cols: List[str]) -> pl.Series:
    """Cost / Conversions"""
    return _safe_div(df[cols[0]], df[cols[1]])

def calc_ctr(df: pl.DataFrame, cols: List[str]) -> pl.Series:
    """Clicks / Impressions"""
    return _safe_div(df[cols[0]], df[cols[1]])

def calc_conversion_rate(df: pl.DataFrame, cols: List[str]) -> pl.Series:
    """Conversions / Clicks (or Sessions)"""
    return _safe_div(df[cols[0]], df[cols[1]])

def calc_profit_margin(df: pl.DataFrame, cols: List[str]) -> pl.Series:
    """(Revenue - Cost) / Revenue"""
    revenue = df[cols[0]]
    cost = df[cols[1]]
    profit = revenue - cost
    return _safe_div(profit, revenue)

# --- Registry ---

DERIVED_METRICS_REGISTRY = {
    "ROAS": DerivedMetricDefinition(
        name="Return on Ad Spend (ROAS)",
        description="Revenue generated per unit of ad spend",
        required_roles=["financial", "financial"],
        required_keywords=[
            ["revenue", "sales", "gross"], 
            ["ad", "marketing", "spend", "cost", "budget"]
        ],
        formula=calc_roas
    ),
    "CPA": DerivedMetricDefinition(
        name="Cost Per Acquisition (CPA)",
        description="Cost to acquire one paying customer/conversion",
        required_roles=["financial", "funnel"], # Cost, Conversions
        required_keywords=[
            ["cost", "spend", "budget"],
            ["convert", "conversion", "acquisition", "sale", "order"]
        ],
        formula=calc_cpa
    ),
    "CTR": DerivedMetricDefinition(
        name="Click-Through Rate (CTR)",
        description="Percentage of impressions that resulted in a click",
        required_roles=["funnel", "funnel"],
        required_keywords=[
            ["click"], 
            ["impression", "view"]
        ],
        formula=calc_ctr
    ),
    "CVR": DerivedMetricDefinition(
        name="Conversion Rate",
        description="Percentage of visitors/clicks that converted",
        required_roles=["funnel", "funnel"],
        required_keywords=[
            ["convert", "conversion", "order", "sale"],
            ["click", "session", "visit"]
        ],
        formula=calc_conversion_rate
    ),
     "Margin": DerivedMetricDefinition(
        name="Profit Margin",
        description="Profit as a percentage of revenue",
        required_roles=["financial", "financial"],
        required_keywords=[
            ["revenue", "sales"],
            ["cost", "expense", "cogs"]
        ],
        formula=calc_profit_margin
    ),
    
    # --- Profitability Metrics ---
    "GrossMargin": DerivedMetricDefinition(
        name="Gross Margin",
        description="(Revenue - COGS) / Revenue",
        required_roles=["financial", "financial"],
        required_keywords=[
            ["revenue", "sales", "income"],
            ["cogs", "cost of goods", "direct cost"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]] - df[cols[1]], df[cols[0]])
    ),
    "NetMargin": DerivedMetricDefinition(
        name="Net Margin",
        description="Net Profit / Revenue",
        required_roles=["financial", "financial"],
        required_keywords=[
            ["net profit", "net income", "profit"],
            ["revenue", "sales"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]], df[cols[1]])
    ),
    "ROI": DerivedMetricDefinition(
        name="Return on Investment",
        description="(Gain - Cost) / Cost",
        required_roles=["financial", "financial"],
        required_keywords=[
            ["revenue", "return", "gain", "sales"],
            ["investment", "cost", "spend"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]] - df[cols[1]], df[cols[1]])
    ),
    
    # --- Efficiency Metrics ---
    "CAC": DerivedMetricDefinition(
        name="Customer Acquisition Cost",
        description="Marketing Spend / New Customers",
        required_roles=["financial", "customer"],
        required_keywords=[
            ["marketing", "ad spend", "acquisition cost"],
            ["customer", "new customer", "acquisition"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]], df[cols[1]])
    ),
    "LTV_CAC": DerivedMetricDefinition(
        name="LTV to CAC Ratio",
        description="Customer Lifetime Value / Customer Acquisition Cost",
        required_roles=["financial", "financial"],
        required_keywords=[
            ["ltv", "lifetime value", "customer value"],
            ["cac", "acquisition cost"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]], df[cols[1]])
    ),
    "InventoryTurnover": DerivedMetricDefinition(
        name="Inventory Turnover",
        description="COGS / Average Inventory",
        required_roles=["financial", "inventory"],
        required_keywords=[
            ["cogs", "cost of goods"],
            ["inventory", "stock"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]], df[cols[1]])
    ),
    
    # --- Growth Metrics ---
    "GrowthRate": DerivedMetricDefinition(
        name="Growth Rate",
        description="(Current - Previous) / Previous",
        required_roles=["financial", "financial"],
        required_keywords=[
            ["current", "new", "recent"],
            ["previous", "old", "prior"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]] - df[cols[1]], df[cols[1]])
    ),
    
    # --- Marketing Metrics ---
    "CPC": DerivedMetricDefinition(
        name="Cost Per Click",
        description="Ad Spend / Clicks",
        required_roles=["financial", "funnel"],
        required_keywords=[
            ["ad spend", "cost", "spend"],
            ["click"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]], df[cols[1]])
    ),
    "BounceRate": DerivedMetricDefinition(
        name="Bounce Rate",
        description="Bounces / Total Sessions",
        required_roles=["funnel", "funnel"],
        required_keywords=[
            ["bounce"],
            ["session", "visit"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]], df[cols[1]])
    ),
    
    # --- Operational Metrics ---
    "FulfillmentRate": DerivedMetricDefinition(
        name="Fulfillment Rate",
        description="Orders Fulfilled / Total Orders",
        required_roles=["funnel", "funnel"],
        required_keywords=[
            ["fulfilled", "completed", "shipped"],
            ["order", "total order"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]], df[cols[1]])
    ),
    "UtilizationRate": DerivedMetricDefinition(
        name="Utilization Rate",
        description="Actual Usage / Capacity",
        required_roles=["inventory", "inventory"],
        required_keywords=[
            ["usage", "used", "actual"],
            ["capacity", "available", "total"]
        ],
        formula=lambda df, cols: _safe_div(df[cols[0]], df[cols[1]])
    )
}

def get_derived_metric(name: str) -> Optional[DerivedMetricDefinition]:
    """Retrieve a metric definition by name (case-insensitive)."""
    name_normalized = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    
    for key, metric in DERIVED_METRICS_REGISTRY.items():
        key_normalized = key.lower().replace(" ", "").replace("_", "").replace("-", "")
        if key_normalized == name_normalized:
            return metric
        
        # Check if name is in metric's full name
        metric_name_normalized = metric.name.lower().replace(" ", "").replace("_", "").replace("-", "")
        if name_normalized in metric_name_normalized or metric_name_normalized in name_normalized:
            return metric
            
    return None
