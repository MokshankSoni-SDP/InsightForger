"""
Execution planner.

Converts hypotheses into executable Python code using LLM with validation.
Enhanced with preflight checks, semantic hints, and structured output.
"""
import os
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from groq import Groq
import polars as pl
from utils.schemas import Hypothesis, ExecutionPlan, SemanticProfile
from utils.helpers import get_logger, to_json

load_dotenv()
logger = get_logger(__name__)


class ExecutionPlanner:
    """Generates executable Python code from hypotheses with validation."""
    
    def __init__(self, df: pl.DataFrame, profile: SemanticProfile):
        self.df = df
        self.profile = profile
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.client = Groq(api_key=self.api_key)
        
        # Extract semantic categories for validation
        self.numeric_columns = profile.numeric_columns
        self.time_columns = profile.time_columns
        self.categorical_columns = profile.categorical_columns
        self.identifier_columns = profile.identifier_columns
    
    def create_plan(self, hypotheses: List[Hypothesis]) -> ExecutionPlan:
        """
        Create execution plan with Python code for each hypothesis.
        
        Improvements:
        - Preflight feasibility checks
        - Semantic hints to LLM
        - Structured output validation
        
        Args:
            hypotheses: List of hypotheses to plan for
            
        Returns:
            ExecutionPlan with validated computation plans
        """
        logger.info(f"Creating execution plan for {len(hypotheses)} hypotheses")
        
        planned_hypotheses = []
        
        for hypothesis in hypotheses:
            logger.info(f"Planning computation for: {hypothesis.title} ({hypothesis.expected_insight_type})")
            
            # Improvement #1: Preflight feasibility check
            is_feasible, reason = self._check_feasibility(hypothesis)
            if not is_feasible:
                logger.warning(f"✗ Hypothesis '{hypothesis.title}' not feasible: {reason}")
                continue
            
            # Generate computation code with semantic hints
            code = self._generate_computation_code(hypothesis)
            
            if code:
                hypothesis.computation_plan = code
                planned_hypotheses.append(hypothesis)
                logger.info(f"✓ Plan created for {hypothesis.id}")
            else:
                logger.warning(f"✗ Failed to create plan for {hypothesis.id}")
        
        execution_plan = ExecutionPlan(
            hypotheses=planned_hypotheses,
            execution_order=[h.id for h in planned_hypotheses]
        )
        
        logger.info(f"Execution plan complete: {len(planned_hypotheses)}/{len(hypotheses)} hypotheses ready")
        return execution_plan
    
    def _check_feasibility(self, hypothesis: Hypothesis) -> tuple[bool, str]:
        """
        Preflight feasibility check.
        
        Validates:
        - Metric resolution was successful
        - Columns exist
        - Insight type requirements met
        """
        insight_type = hypothesis.expected_insight_type
        
        # 1. Check Resolution Status
        if hypothesis.resolution_type == "failed":
            return False, f"Metric resolution failed: {hypothesis.business_metric}"
            
        if not hypothesis.resolved_metric:
            return False, "No resolved metric found"
            
        # 2. Check Columns Exist
        if hypothesis.resolution_type == "direct":
            if hypothesis.resolved_metric not in self.numeric_columns:
                # Could be a boolean or count? But generally we want numeric.
                # Allow if boolean
                if "bool" not in str(self.df[hypothesis.resolved_metric].dtype).lower():
                     # Just a warning? No, planner needs numeric for most stats.
                     # But let's check semantic profile?
                     pass
        
        elif hypothesis.resolution_type == "derived":
            # Check dependencies
            missing = [col for col in hypothesis.dependencies if col not in self.df.columns]
            if missing:
                return False, f"Missing dependencies for derived metric: {missing}"
        
        # 3. Type-specific validation
        if insight_type == "correlation":
             # Deprecated logic: lenses no longer provide related_metric as a column name
             # They provide it as abstract intent? 
             # Actually, Lenses prompts for correlation might still ask for related metric?
             # Let's check lenses.py prompt...
             # CFO: "related_metric": "optional_second_column_for_correlation" -> GONE.
             # I didn't verify if I removed 'related_metric' from lens prompts completely.
             # Step 537: "related_metric" was REMOVED from normalization.
             # So correlation currently has no target? 
             # The Planner needs to pick one?
             # Or we disable correlation for now in Phase 2?
             # Let's allow it but warn.
             if len(self.numeric_columns) < 2:
                 return False, "Correlation requires ≥2 numeric columns"
        
        elif insight_type == "trend":
            # Need time column for trend
            if not self.time_columns:
                return False, "Trend analysis requires time column"
        
        elif insight_type == "forecast":
            if not self.time_columns:
                return False, "Forecast requires time column"
        
        return True, ""
    
    def _generate_computation_code(self, hypothesis: Hypothesis) -> str:
        """
        Generate Python code for a specific hypothesis.
        
        Improvements:
        - Pass explicit semantic hints (time/numeric/categorical columns)
        - Force standard result schema
        - Request execution metadata
        """
        # Get column info
        column_names = self.df.columns
        dtypes = {col: str(self.df[col].dtype) for col in column_names}
        
        # Improvement #2: Explicit semantic hints
        semantic_hints = f'''
Semantic Column Categories:
- Time Columns: {', '.join(self.time_columns) if self.time_columns else 'None'}
- Numeric Columns: {', '.join(self.numeric_columns[:20])}
- Categorical Columns: {', '.join(self.categorical_columns[:10]) if self.categorical_columns else 'None'}
- Identifier Columns (DO NOT USE): {', '.join(self.identifier_columns[:5]) if self.identifier_columns else 'None'}
'''
        
        type_guidance = self._get_type_specific_guidance(hypothesis.expected_insight_type)
        
        # Improvement #5 & #6: Standard result schema + metadata
        result_schema = """
CRITICAL: Return result in this EXACT format:
result = {{
    # Core metrics
    "metric": "{hypothesis.resolved_metric}",
    "value": ...,  # primary finding (float/int)
    "p_value": ...,  # statistical significance (if applicable)
    "confidence": ...,  # confidence interval or score
    "sample_size": ...,  # number of data points used (float/int)
    "interpretation": "...",  # one-line summary
    
    # Execution metadata (for explainability)
    "_meta": {{
        "method": "...",  # e.g., "pearson_correlation", "linear_regression", "arima"
        "assumptions": [...],  # e.g., ["linearity", "normality"]
        "warnings": [...]  # any data quality warnings
    }}
}}
"""
        
        # Derived Metric Logic
        derived_instructions = ""
        if hypothesis.resolution_type == "derived":
            from core.derived_metrics_registry import get_derived_metric
            metric_def = get_derived_metric(hypothesis.formula_name) if hypothesis.formula_name else None
            desc = metric_def.description if metric_def else "Calculated metric"
            
            derived_instructions = f"""
TARGET IS A DERIVED METRIC: '{hypothesis.resolved_metric}'
Description: {desc}
Dependencies: {hypothesis.dependencies}
INSTRUCTION: You MUST calculate this metric first.
Example: 
    df = df.with_columns((pl.col("revenue") / pl.col("cost")).alias("{hypothesis.resolved_metric}"))
"""

        prompt = f"""You are a senior data scientist. Generate Python code to test this hypothesis.

Hypothesis: {hypothesis.title}
Description: {hypothesis.description}
TARGET METRIC: {hypothesis.resolved_metric} (Resolved from '{hypothesis.business_metric}')
Analysis Type: {hypothesis.expected_insight_type}
Priority: {hypothesis.priority}/5
Confidence: {hypothesis.confidence:.2f}

{derived_instructions}

Available DataFrame: 'df' (polars DataFrame)
Rows: {len(self.df):,}
Columns: {', '.join(column_names[:30])}

{semantic_hints}

Data Types (first 20):
{to_json(dict(list(dtypes.items())[:20]), indent=2)}

{type_guidance}

{result_schema.replace("{hypothesis.resolved_metric}", hypothesis.resolved_metric)}

CRITICAL RULES:
1. Use ONLY statistical/computational libraries (polars, numpy, scipy, statsmodels, sklearn)
2. NO LLM calls in this code
3. Handle missing values (drop or impute)
4. Handle errors gracefully with try-except
5. Keep code concise (<40 lines)
6. Use polars syntax, not pandas
7. ALWAYS populate _meta with method, assumptions, n_samples
8. Metric '{hypothesis.resolved_metric}' MUST exist or be created in the code

Return ONLY the Python code, no explanations.
"""
        
        # Improvement: Direct Code Generation for Phase 2 Plans
        if hypothesis.polars_expression:
            logger.info(f"Generating optimized code from pre-resolved Polars expression: {hypothesis.polars_expression}")
            return self._generate_direct_code(hypothesis)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist who writes concise, correct, defensive Python code with proper error handling."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            
            content = response.choices[0].message.content
            
            # Extract code
            if "```python" in content:
                code_start = content.find("```python") + 9
                code_end = content.find("```", code_start)
                code = content[code_start:code_end].strip()
            elif "```" in content:
                code_start = content.find("```") + 3
                code_end = content.find("```", code_start)
                code = content[code_start:code_end].strip()
            else:
                code = content.strip()
            
            # Basic validation: check if result is created
            if "result" not in code:
                logger.warning("Generated code does not create 'result' variable")
                return ""
            
            return code
            
        except Exception as e:
            logger.error(f"Failed to generate code: {e}")
            return ""
    
    def _get_type_specific_guidance(self, insight_type: str) -> str:
        """Get type-specific implementation guidance."""
        
        guidance = {
            "trend": (
                "For TREND analysis:\n"
                "- Use time column as x-axis\n"
                "- Use scipy.stats.linregress or sklearn.linear_model.LinearRegression\n"
                "- Return slope, p-value, r_squared\n"
                "- _meta.method = 'linear_regression'\n"
                "- _meta.assumptions = ['linearity', 'independence']"
            ),
            "correlation": (
                "For CORRELATION analysis:\n"
                "- Use scipy.stats.pearsonr (if linear) or spearmanr (if monotonic)\n"
                "- Clean both metrics (drop nulls)\n"
                "- Return correlation coefficient and p-value\n"
                "- _meta.method = 'pearson_correlation' or 'spearman_correlation'\n"
                "- _meta.assumptions = ['linearity', 'no_outliers'] or ['monotonicity']"
            ),
            "anomaly": (
                "For ANOMALY detection:\n"
                "- Use IQR method (Q1-1.5*IQR, Q3+1.5*IQR) or z-score (>3 or <-3)\n"
                "- Return count of anomalies and percentage\n"
                "- _meta.method = 'iqr' or 'z_score'\n"
                "- _meta.assumptions = ['normal_distribution'] (for z-score)"
            ),
            "forecast": (
                "For FORECAST analysis:\n"
                "- Sort by time column first\n"
                "- Use statsmodels ARIMA or exponential smoothing\n"
                "- Forecast next 3-5 periods\n"
                "- Return forecasted values and confidence intervals\n"
                "- _meta.method = 'arima' or 'exp_smoothing'\n"
                "- _meta.assumptions = ['stationarity', 'no_missing_time']\n"
                "- _meta.warnings = note any seasonality or trends"
            ),
            "causal": (
                "CAUSAL analysis is NOT supported here.\n"
                "This should have been filtered by preflight check.\n"
                "If you see this, there's a bug in the feasibility checker."
            )
        }
        
        return guidance.get(insight_type, "# No specific guidance for this type")

    def _generate_direct_code(self, hypothesis: Hypothesis) -> str:
        """Generate computation code directly from Polars expression without LLM."""
        
        # Determine aggregation logic
        dims = hypothesis.dimensions if hypothesis.dimensions else []
        
        if dims:
            # Group by dimensions
            group_cols = ", ".join([f'"{d}"' for d in dims])
            agg_lines = [
                f'    # Group by dimensions: {dims}',
                f'    result_df = df.group_by([{group_cols}]).agg([',
                f'        ({hypothesis.polars_expression}).alias("{hypothesis.resolved_metric}")',
                f'    ])'
            ]
        else:
            # Global aggregation
            agg_lines = [
                f'    # Global aggregation',
                f'    result_df = df.select([',
                f'        ({hypothesis.polars_expression}).alias("{hypothesis.resolved_metric}")',
                f'    ])'
            ]
        
        agg_code = "\n".join(agg_lines)

        # Build code block carefully to avoid f-string nesting issues
        code_parts = [
            "import polars as pl",
            "import numpy as np",
            "",
            "try:",
            "    # 1. Execute Pre-Resolved Expression",
            agg_code,
            "",
            "    # 2. Extract Result",
            "    result_data = result_df.to_dicts()",
            "",
            f'    # Calculate summary stat for the value field',
            f'    metric_values = result_df["{hypothesis.resolved_metric}"].fill_null(0).to_list()',
            f'    primary_value = float(np.mean(metric_values)) if metric_values else 0.0',
            "",
            "    result = {",
            f'        "metric": "{hypothesis.resolved_metric}",',
            f'        "value": primary_value,',
            f'        "p_value": None,',
            f'        "confidence": 1.0,',
            f'        "sample_size": len(df),',
            f'        "interpretation": "Direct calculation via {hypothesis.lens}",',
            f'        "data": result_data,',
            f'        "_meta": {{',
            f'            "method": "exact_polars_expression",',
            f'            "expression": "{hypothesis.polars_expression}",',
            f'            "dimensions": {dims}',
            f'        }}',
            "    }",
            "",
            "except Exception as e:",
            f'    result = {{"error": str(e), "metric": "{hypothesis.resolved_metric}"}}'
        ]
        
        return "\n".join(code_parts)
