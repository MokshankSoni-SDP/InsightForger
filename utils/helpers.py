"""
Utility helper functions used across the pipeline.
Enhanced with improved type inference, data quality warnings, and performance optimizations.
"""
import os
import hashlib
import logging
from typing import Any, Dict, List, Tuple, Union
import json
from datetime import datetime, date
import polars as pl
import numpy as np
from difflib import SequenceMatcher  # For fuzzy matching acronyms

# ============================================================================
# BUSINESS ACRONYM DICTIONARY (E-commerce/Marketing Awareness)
# ============================================================================
BUSINESS_ACRONYM_MAP = {
    # Abandoned/Cart metrics
    "abd": "abandoned", "aband": "abandoned",
    # Advertising/Marketing metrics
    "cpc": "cost_per_click", "cpm": "cost_per_mille", "cpa": "cost_per_acquisition",
    "impr": "impressions", "imp": "impressions", "conv": "conversions",
    "cvr": "conversion_rate", "ctr": "click_through_rate",
    "roas": "return_on_ad_spend", "roi": "return_on_investment",
    # Customer metrics
    "aov": "average_order_value", "ltv": "lifetime_value",
    "clv": "customer_lifetime_value", "cac": "customer_acquisition_cost",
    "arpu": "average_revenue_per_user",
    # Engagement metrics
    "sess": "sessions", "vis": "visitors", "uv": "unique_visitors", "pv": "page_views",
    # E-commerce metrics
    "gmv": "gross_merchandise_value", "aur": "average_unit_retail",
    "asp": "average_selling_price", "sku": "stock_keeping_unit", "qty": "quantity",
    # Time-based
    "yoy": "year_over_year", "mom": "month_over_month",
    "wow": "week_over_week", "dod": "day_over_day",
    # Other common abbreviations
    "rev": "revenue", "prof": "profit", "marg": "margin",
    "disc": "discount", "ret": "return", "refund": "refund",
}

# Note: Logging configuration should be done once in main.py
# This just provides a getter for modules to use

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    return logging.getLogger(name)


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_hash = hashlib.md5(os.urandom(16)).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{random_hash}" if prefix else f"{timestamp}_{random_hash}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def normalize_column_name(name: str) -> str:
    """
    Normalize column name to snake_case.
    
    Args:
        name: Original column name
        
    Returns:
        Normalized column name
    """
    import re
    # Convert to lowercase
    name = name.lower()
    # Replace spaces and special chars with underscore
    name = re.sub(r'[^\w\s]', '_', name)
    name = re.sub(r'\s+', '_', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name


def has_gpu() -> bool:
    """
    Check if GPU is available.
    
    Note: Marked for FUTURE USE - not currently wired into forecasting/anomaly engines.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """
    Get the appropriate device (cuda or cpu).
    
    Note: Marked for FUTURE USE - not currently wired into forecasting/anomaly engines.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def calculate_basic_stats(series: pl.Series) -> Dict[str, Any]:
    """
    Calculate basic statistics for a series.
    
    Args:
        series: Polars series
        
    Returns:
        Dictionary with basic stats
    """
    stats = {}
    
    try:
        if series.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            # Numeric stats
            stats["min"] = series.min()
            stats["max"] = series.max()
            stats["mean"] = series.mean()
            stats["median"] = series.median()
            stats["std"] = series.std()
        
        stats["null_count"] = series.null_count()
        stats["unique_count"] = series.n_unique()
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Error calculating stats: {e}")
    
    return stats


# ============================================================================
# INTELLIGENT PROFILING FUNCTIONS
# ============================================================================

def detect_ordinal_rank(series: pl.Series) -> Tuple[bool, float]:
    """
    Detect if a numeric column is an ordinal rank/position using Sequential Integer Density.
    
    Logic:
    1. Check if dtype is Integer
    2. Calculate max_value vs unique_count (sequential density)
    3. Perform sum test for perfect sequences
    
    Args:
        series: Polars series to analyze
        
    Returns:
        Tuple of (is_ordinal, confidence)
        - is_ordinal: True if likely an ordinal dimension
        - confidence: 0.0-1.0 confidence score
    """
    logger = get_logger(__name__)
    
    # Must be integer type
    dtype = str(series.dtype).lower()
    if "int" not in dtype:
        return False, 0.0
    
    try:
        # Get non-null values
        values = series.drop_nulls()
        if len(values) < 2:
            return False, 0.0
        
        unique_count = values.n_unique()
        max_value = values.max()
        min_value = values.min()
        
        # Check for negative values (ranks/positions are usually positive)
        if min_value < 0:
            return False, 0.0
        
        # Test 1: Sequential Density (max ≈ unique_count)
        # If max is very close to unique_count, it's likely a sequence
        if max_value == 0:
            return False, 0.0
            
        density_ratio = unique_count / max_value
        
        # If density >= 90%, likely ordinal
        if density_ratio >= 0.9:
            # Test 2: Perfect Sequence (sum test)
            # For a perfect sequence 1,2,3,...,n: sum = n*(n+1)/2
            expected_range = max_value - min_value + 1
            expected_sum = (expected_range * (expected_range + 1)) / 2
            
            # Adjust for min_value offset
            if min_value > 0:
                expected_sum += min_value * unique_count - (min_value * (min_value + 1)) / 2
            
            actual_sum = values.sum()
            
            if expected_sum > 0:
                sum_ratio = abs(actual_sum - expected_sum) / expected_sum
                
                # Perfect or near-perfect sequence
                if sum_ratio < 0.1:  # Within 10%
                    logger.debug(f"Detected ordinal rank in '{series.name}': density={density_ratio:.2f}, sum_test={1-sum_ratio:.2f}")
                    return True, 0.99  # Very high confidence
                elif sum_ratio < 0.3:  # Within 30%
                    return True, 0.85  # High confidence
            
            # High density but not perfect sequence (shuffled ranks)
            return True, 0.75
        
        # Medium density (70-90%) - possible ordinal
        elif density_ratio >= 0.7:
            return True, 0.60
        
        return False, 0.0
        
    except Exception as e:
        logger.debug(f"Error in ordinal detection: {e}")
        return False, 0.0


def expand_acronyms(col_name: str, acronym_map: Dict[str, str] = None) -> str:
    """
    Expand business acronyms in column name using dictionary.
    
    Args:
        col_name: Column name to expand
        acronym_map: Dictionary of acronyms (default: BUSINESS_ACRONYM_MAP)
        
    Returns:
        Expanded column name
    """
    if acronym_map is None:
        acronym_map = BUSINESS_ACRONYM_MAP
    
    col_lower = col_name.lower()
    
    # Split by common separators
    parts = col_lower.replace("_", " ").replace("-", " ").split()
    
    # Expand each part
    expanded_parts = []
    for part in parts:
        if part in acronym_map:
            expanded_parts.append(acronym_map[part])
        else:
            expanded_parts.append(part)
    
    expanded = "_".join(expanded_parts)
    return expanded


def fuzzy_match_semantic(col_name: str, known_patterns: List[str], threshold: float = 0.7) -> Tuple[str, float]:
    """
    Fuzzy match column name to known semantic patterns.
    
    Args:
        col_name: Column name to match
        known_patterns: List of known patterns to match against
        threshold: Minimum similarity threshold (0.0-1.0)
        
    Returns:
        Tuple of (best_match, similarity_score)
    """
    col_lower = col_name.lower()
    best_match = ""
    best_score = 0.0
    
    for pattern in known_patterns:
        similarity = SequenceMatcher(None, col_lower, pattern.lower()).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = pattern
    
    if best_score >= threshold:
        return best_match, best_score
    else:
        return "", 0.0


def validate_zeros(series: pl.Series, df: pl.DataFrame, semantic_role: str) -> Tuple[bool, str]:
    """
    Validate if zeros in a column are business reality vs missing data.
    
    Uses Zero-to-Null Ratio and Trigger Column Correlation.
    
    Args:
        series: Series to validate
        df: Full dataframe for trigger column analysis
        semantic_role: Semantic role of the column
        
    Returns:
        Tuple of (is_validated, reason)
        - is_validated: True if zeros are valid business data
        - reason: Explanation of validation
    """
    logger = get_logger(__name__)
    
    try:
        # Count zeros and nulls
        zero_count = (series == 0).sum()
        null_count = series.null_count()
        total_count = len(series)
        
        zero_ratio = zero_count / total_count if total_count > 0 else 0
        
        # If very few zeros, no need to validate
        if zero_ratio < 0.3:
            return True, "Low zero ratio - not a concern"
        
        # Test 1: Zero-to-Null Ratio
        # If we have 5x more zeros than nulls, zeros are likely valid data
        if null_count > 0:
            zero_to_null_ratio = zero_count / null_count
            if zero_to_null_ratio > 5:
                return True, f"Zero-to-null ratio ({zero_to_null_ratio:.1f}:1) indicates zeros are valid data"
        elif zero_count > 0 and null_count == 0:
            # No nulls at all - zeros are definitely intentional
            return True, "No nulls present - zeros are intentional data points"
        
        # Test 2: Trigger Column Correlation (for financial/funnel metrics)
        if semantic_role in ["financial", "funnel"]:
            # Find potential trigger columns
            trigger_keywords = ["click", "view", "visit", "session", "impression", "traffic", "user"]
            trigger_cols = [
                col for col in df.columns
                if any(kw in col.lower() for kw in trigger_keywords)
                and col != series.name
            ]
            
            for trigger_col in trigger_cols:
                try:
                    trigger_series = df[trigger_col]
                    
                    # Check correlation: when trigger=0, metric should be 0
                    trigger_zeros = (trigger_series == 0)
                    metric_zeros = (series == 0)
                    
                    # Count cases where both are zero
                    both_zero = (trigger_zeros & metric_zeros).sum()
                    trigger_zero_count = trigger_zeros.sum()
                    
                    if trigger_zero_count > 0:
                        correlation = both_zero / trigger_zero_count
                        
                        if correlation > 0.8:  # 80% correlation
                            return True, f"Zeros validated by trigger column '{trigger_col}' (correlation: {correlation:.1%})"
                
                except Exception:
                    continue
        
        # Default: zeros not validated but not necessarily invalid
        return False, f"High zero ratio ({zero_ratio:.1%}) - unable to validate with trigger columns"
        
    except Exception as e:
        logger.debug(f"Error in zero validation: {e}")
        return False, "Validation error"


def infer_statistical_type(series: pl.Series, unique_ratio: float) -> str:
    """
    Infer statistical type of column (fact-based).
    
    Improvements:
    - Boolean detection requires explicit dtype or strong name hint
    - Identifier detection excludes numeric columns and requires name hints
    - Ordinal rank detection for sequential integers
    """
    dtype = str(series.dtype).lower()
    col_name = series.name.lower()
    
    # Temporal
    if "date" in dtype or "time" in dtype:
        return "temporal"
    
    # Boolean (explicit dtype check first)
    if series.dtype == pl.Boolean:
        return "boolean"
    
    # Boolean (weak signal - only if name strongly suggests it)
    if unique_ratio < 0.01 and series.n_unique() <= 2:
        if any(kw in col_name for kw in ["is_", "has_", "flag", "active", "enabled"]):
            return "boolean"
    
    # Identifier (improved logic - exclude numeric and require name hints)
    if unique_ratio > 0.95:
        # Exclude numeric columns (timestamps, metrics can have high cardinality)
        if "int" not in dtype and "float" not in dtype:
            # Require name hint for identifier
            if any(kw in col_name for kw in ["id", "key", "uuid", "hash", "code"]):
                return "identifier"
        # Even for numeric, if name says ID, trust it
        elif any(kw in col_name for kw in ["id", "_id", "key"]):
            return "identifier"
    
    # Numeric - but check for ordinal ranks first
    if "int" in dtype or "float" in dtype:
        # Check if it's an ordinal rank (position, rank, order)
        is_ordinal, ordinal_conf = detect_ordinal_rank(series)
        if is_ordinal and ordinal_conf >= 0.75:
            # Check name hints for ordinal
            if any(kw in col_name for kw in ["position", "rank", "order", "place", "level", "tier", "grade"]):
                return "categorical"  # Treat ordinal as categorical dimension
        
        return "numeric"
    
    # Categorical
    if "str" in dtype or "utf8" in dtype:
        return "categorical"
    
    return "other"


def infer_semantic_role(series: pl.Series, unique_ratio: float, statistical_type: str, df: pl.DataFrame = None) -> Tuple[str, float]:
    """
    Infer semantic role with confidence (interpretation-based).
    
    Enhanced with:
    - Acronym expansion for better matching
    - Fuzzy matching for similar patterns
    - Neighbor validation (columns in same row)
    
    Args:
        series: Column to analyze
        unique_ratio: Ratio of unique values
        statistical_type: Statistical type from infer_statistical_type
        df: Optional full dataframe for neighbor validation
    
    Returns:
        Tuple of (semantic_role, confidence)
    """
    col_name = series.name.lower()
    
    # Expand acronyms first
    expanded_name = expand_acronyms(col_name)
    
    # Initialize confidence scores
    scores = {
        "time": 0.0,
        "product": 0.0,
        "financial": 0.0,
        "inventory": 0.0,
        "funnel": 0.0,
        "customer": 0.0,
        "identifier": 0.0,
        "other": 0.3  # Baseline for unknown
    }
    
    # 1. Statistical Type Priors
    if statistical_type == "temporal":
        scores["time"] += 0.7
    elif statistical_type == "categorical":
        scores["product"] += 0.2
        scores["customer"] += 0.2
    elif statistical_type == "numeric":
        scores["financial"] += 0.2
        scores["inventory"] += 0.2
        scores["funnel"] += 0.1
    elif statistical_type == "identifier":
        scores["identifier"] += 0.6
        scores["customer"] += 0.1
    
    # 2. Cardinality Intelligence
    if unique_ratio > 0.9:
        scores["identifier"] += 0.1
    elif unique_ratio < 0.1:
        scores["financial"] += 0.1
        scores["funnel"] += 0.1
    
    # 3. Keyword Boosting (use both original and expanded names)
    priors = [
        # Financial
        ("price", "financial", 0.25), ("cost", "financial", 0.25), ("amount", "financial", 0.15),
        ("sales", "financial", 0.25), ("revenue", "financial", 0.25), ("profit", "financial", 0.25),
        ("margin", "financial", 0.20), ("tax", "financial", 0.15), ("budget", "financial", 0.25),
        
        # Inventory / Quantity
        ("qty", "inventory", 0.25), ("quantity", "inventory", 0.25), ("stock", "inventory", 0.25),
        ("units", "inventory", 0.15), ("count", "inventory", 0.15),
        
        # Customer / User
        ("customer", "customer", 0.25), ("client", "customer", 0.25), ("user", "customer", 0.20),
        ("account", "customer", 0.15),
        
        # Product
        ("product", "product", 0.25), ("item", "product", 0.20), ("sku", "product", 0.25),
        ("category", "product", 0.15), ("brand", "product", 0.15),
        
        # Time
        ("date", "time", 0.25), ("time", "time", 0.25), ("year", "time", 0.20),
        
        # Funnel / Marketing
        ("click", "funnel", 0.25), ("view", "funnel", 0.20), ("impression", "funnel", 0.20),
        ("convert", "funnel", 0.25), ("conversion", "funnel", 0.25),
        ("session", "funnel", 0.15), ("abandoned", "funnel", 0.25), ("cart", "funnel", 0.15),
        
        # Identifier
        ("id", "identifier", 0.25), ("key", "identifier", 0.20),
    ]
    
    # Check both original and expanded names
    for fragment, category, boost in priors:
        if fragment in col_name or fragment in expanded_name:
            scores[category] += boost
    
    # 4. Fuzzy Matching for Expanded Names
    # Check if expanded name matches known patterns
    funnel_patterns = ["abandoned_carts", "click_through_rate", "conversion_rate", "impressions"]
    financial_patterns = ["cost_per_click", "return_on_ad_spend", "average_order_value"]
    
    best_funnel_match, funnel_score = fuzzy_match_semantic(expanded_name, funnel_patterns, threshold=0.7)
    if funnel_score >= 0.7:
        scores["funnel"] += 0.45  # Significant boost for fuzzy match
    
    best_financial_match, financial_score = fuzzy_match_semantic(expanded_name, financial_patterns, threshold=0.7)
    if financial_score >= 0.7:
        scores["financial"] += 0.45
    
    # 5. Neighbor Validation (if dataframe provided)
    if df is not None:
        # Check if column is in a "performance neighborhood"
        performance_keywords = ["sales", "revenue", "price", "cost", "profit"]
        neighbor_cols = [c for c in df.columns if c != series.name]
        performance_neighbors = sum(1 for c in neighbor_cols if any(kw in c.lower() for kw in performance_keywords))
        
        if performance_neighbors >= 2:
            # In a performance neighborhood - boost financial/funnel confidence
            if scores["financial"] > 0.3 or scores["funnel"] > 0.3:
                scores["financial"] *= 1.2
                scores["funnel"] *= 1.2
    
    # 6. Normalize and Pick Winner
    for k in scores:
        scores[k] = min(0.95, scores[k])
    
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]
    
    # 7. Sanity Checks
    if best_category == "time" and statistical_type != "temporal" and best_score < 0.8:
        scores["time"] = 0
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
    
    return best_category, round(best_score, 2)


def infer_distribution_type(series: pl.Series, statistical_type: str, df: pl.DataFrame = None, semantic_role: str = None) -> Tuple[str, bool, str]:
    """
    Infer distribution shape for numeric columns with zero validation.
    
    Improvements:
    - Samples large columns (>5000 rows) for performance
    - Validates zeros using zero-to-null ratio and trigger columns
    
    Args:
        series: Series to analyze
        statistical_type: Statistical type
        df: Optional full dataframe for zero validation
        semantic_role: Optional semantic role for context
    
    Returns:
        Tuple of (distribution_type, sparsity_validated, validation_reason)
    """
    if statistical_type != "numeric":
        return "unknown", False, ""
    
    try:
        values = series.drop_nulls().to_numpy()
        
        if len(values) < 10:
            return "unknown", False, ""
        
        # Performance improvement: Sample large arrays
        if len(values) > 5000:
            np.random.seed(42)  # Reproducibility
            values = np.random.choice(values, 5000, replace=False)
        
        # Check for sparsity (mostly zeros)
        zero_ratio = np.sum(values == 0) / len(values)
        
        if zero_ratio > 0.7:
            # High zero ratio - validate if zeros are business reality
            if df is not None and semantic_role is not None:
                is_validated, reason = validate_zeros(series, df, semantic_role)
                return "sparse", is_validated, reason
            else:
                return "sparse", False, "High zero ratio but unable to validate"
        
        # Calculate skewness and kurtosis
        from scipy.stats import skew, kurtosis
        
        skewness = skew(values)
        kurt = kurtosis(values)
        
        # Classify distribution
        if abs(skewness) < 0.5 and abs(kurt) < 1:
            return "normal", False, ""
        elif abs(skewness) >= 2 or kurt > 5:
            return "long_tail", False, ""
        elif abs(skewness) >= 0.5:
            return "skewed", False, ""
        else:
            return "uniform", False, ""
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.debug(f"Error in distribution inference: {e}")
        return "unknown", False, ""


def calculate_basic_stats(series: pl.Series) -> Dict[str, Any]:
    """Calculate basic statistics for a column."""
    stats = {
        "null_count": series.null_count(),
        "null_percentage": series.null_count() / len(series) * 100 if len(series) > 0 else 0,
    }
    
    # Add numeric stats if applicable
    if series.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
        try:
            stats["mean"] = series.mean()
            stats["median"] = series.median()
            stats["std"] = series.std()
            stats["min"] = series.min()
            stats["max"] = series.max()
        except:
            pass
    
    return stats


class RobustJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime, date, and numpy types."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return super().default(obj)
        except:
            return str(obj)


def to_json(data: Any, indent: int = None) -> str:
    """
    Safe JSON serialization wrapper.
    Use this instead of json.dumps for robustness.
    """
    return json.dumps(data, cls=RobustJSONEncoder, indent=indent)
