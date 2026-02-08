"""
Phase 3: Simplified Code Generation Templates

Provides aggregation templates for GLOBAL, DIMENSIONAL, and TEMPORAL scopes.
Simplified version without complex type casting.
"""
from typing import List, Optional


class AggregationTemplates:
    """
    Triple-Lock code generation templates (simplified).
    
    Lock 1: Pre-flight sanitization (filter nulls/zeros only)
    Lock 2: Dimensional boilerplate (aggregation strategy)
    Lock 3: Standardized result wrapper
    """
    
    @staticmethod
    def get_dynamic_imports(analysis_type: str = "ratio") -> str:
        """Generate dynamic imports based on analysis type."""
        base_imports = """import polars as pl
import numpy as np
import traceback"""
        
        if analysis_type in ["correlation", "significance"]:
            base_imports += "\nfrom scipy.stats import pearsonr"
        
        return base_imports
    
    @staticmethod
    def get_sanitization_code(
        columns: List[str],
        denominator_col: Optional[str] = None
    ) -> str:
        """Generate pre-flight data sanitization code (Lock 1) - SIMPLIFIED."""
        sanitization = """# === PRE-FLIGHT SANITIZATION (Lock 1) ===
# Filter invalid data (nulls and zeros in denominators)
"""
        
        # Build filter conditions
        filters = []
        for col in columns:
            filters.append(f"pl.col('{col}').is_not_null()")
        
        if denominator_col:
            filters.append(f"pl.col('{denominator_col}') > 0")
        
        if filters:
            filter_expr = " & ".join(f"({f})" for f in filters)
            sanitization += f"""sanitized_df = df.filter({filter_expr})
"""
        else:
            sanitization += """sanitized_df = df
"""
        
        return sanitization
    
    @staticmethod
    def get_global_template(polars_expression: str) -> str:
        """Generate GLOBAL scope aggregation template (Lock 2)."""
        return f"""# === COMPUTATION (GLOBAL Scope) ===
result_df = sanitized_df.select([
    ({polars_expression}).alias('value')
])

# Extract scalar value
computed_value = result_df['value'][0] if len(result_df) > 0 else None
sample_size = len(sanitized_df)
"""
    
    @staticmethod
    def get_dimensional_template(
        polars_expression: str,
        dimensions: List[str]
    ) -> str:
        """Generate DIMENSIONAL scope aggregation template (Lock 2)."""
        dims_str = str(dimensions)
        
        return f"""# === COMPUTATION (DIMENSIONAL Scope) ===
result_df = sanitized_df.group_by({dims_str}).agg([
    ({polars_expression}).alias('value'),
    pl.count().alias('sample_size')
])

# Sort by value descending to get top segments
result_df = result_df.sort('value', descending=True)

# Extract top value and total sample size
computed_value = result_df['value'][0] if len(result_df) > 0 else None
sample_size = result_df['sample_size'].sum()
top_segment = {{dim: result_df[dim][0] for dim in {dims_str}}} if len(result_df) > 0 else None
"""
    
    @staticmethod
    def get_temporal_template(
        polars_expression: str,
        time_column: str,
        time_grain: str
    ) -> str:
        """Generate TEMPORAL scope aggregation template (Lock 2)."""
        grain_map = {
            "daily": "1d",
            "weekly": "1w",
            "monthly": "1mo",
            "quarterly": "3mo",
            "yearly": "1y"
        }
        interval = grain_map.get(time_grain.lower(), "1d")
        
        return f"""# === COMPUTATION (TEMPORAL Scope) ===
result_df = sanitized_df.sort('{time_column}').group_by_dynamic(
    '{time_column}',
    every='{interval}'
).agg([
    ({polars_expression}).alias('value'),
    pl.count().alias('sample_size')
])

# Calculate trend (simple linear regression)
if len(result_df) > 1:
    x = np.arange(len(result_df))
    y = result_df['value'].to_numpy()
    mask = ~np.isnan(y)
    if mask.sum() > 1:
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        trend_direction = 'increasing' if slope > 0 else 'decreasing'
    else:
        slope = 0
        trend_direction = 'flat'
else:
    slope = 0
    trend_direction = 'flat'

computed_value = result_df['value'][-1] if len(result_df) > 0 else None
sample_size = result_df['sample_size'].sum()
"""
    
    @staticmethod
    def get_result_wrapper(
        aggregation_scope: str,
        dimensions: Optional[List[str]] = None,
        formula: str = "",
        include_trend: bool = False
    ) -> str:
        """Generate standardized result wrapper (Lock 3)."""
        base_result = f"""# === RESULT STANDARDIZATION (Lock 3) ===
result = {{
    'value': float(computed_value) if computed_value is not None else None,
    'sample_size': int(sample_size),
    'is_significant': sample_size >= 30,
    'metadata': {{
        'aggregation_scope': '{aggregation_scope}',
        'dimensions': {dimensions if dimensions else []},
        'formula': '''{formula}'''
    }}
}}
"""
        
        if aggregation_scope == "DIMENSIONAL":
            base_result = base_result.replace(
                "    'metadata': {",
                "    'top_segment': top_segment,\n    'metadata': {"
            )
        
        if include_trend:
            base_result = base_result.replace(
                "    'metadata': {",
                "    'trend_direction': trend_direction,\n    'trend_slope': float(slope),\n    'metadata': {"
            )
        
        return base_result
    
    @staticmethod
    def generate_complete_code(
        polars_expression: str,
        aggregation_scope: str,
        dimensions: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        time_grain: Optional[str] = None,
        columns_used: Optional[List[str]] = None,
        denominator_col: Optional[str] = None,
        analysis_type: str = "ratio"
    ) -> str:
        """Generate complete execution code with Triple-Lock system."""
        imports = AggregationTemplates.get_dynamic_imports(analysis_type)
        
        if not columns_used:
            columns_used = []
        sanitization = AggregationTemplates.get_sanitization_code(
            columns_used, 
            denominator_col
        )
        
        if aggregation_scope == "GLOBAL":
            computation = AggregationTemplates.get_global_template(polars_expression)
        elif aggregation_scope == "DIMENSIONAL":
            computation = AggregationTemplates.get_dimensional_template(
                polars_expression, 
                dimensions or []
            )
        elif aggregation_scope == "TEMPORAL":
            computation = AggregationTemplates.get_temporal_template(
                polars_expression,
                time_column or "Date",
                time_grain or "daily"
            )
        else:
            raise ValueError(f"Unknown aggregation scope: {aggregation_scope}")
        
        result_wrapper = AggregationTemplates.get_result_wrapper(
            aggregation_scope,
            dimensions,
            polars_expression,
            include_trend=(aggregation_scope == "TEMPORAL")
        )
        
        complete_code = f"""{imports}

{sanitization}

{computation}

{result_wrapper}
"""
        
        return complete_code
