"""
Phase 3: Triple-Lock Execution Planner

Generates robust, self-healing Python code from Phase 2 execution plans.
Implements the Triple-Lock system:
- Lock 1: Pre-flight data sanitization
- Lock 2: Expression-aware aggregation (GLOBAL/DIMENSIONAL/TEMPORAL)
- Lock 3: Standardized result wrapper with statistical safety rails
"""
import polars as pl
from typing import List, Optional
from dataclasses import dataclass
from execution.code_templates import AggregationTemplates
from utils.schemas import ExecutionPlan as Phase2ExecutionPlan
from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class Phase3ExecutionPlan:
    """
    Phase 3 execution plan with generated code.
    """
    hypothesis_id: str
    hypothesis_title: str
    generated_code: str
    aggregation_scope: str
    dimensions: List[str]
    polars_expression: str
    columns_used: List[str]


class TripleLockPlanner:
    """
    Generates execution code using Triple-Lock system.
    
    Takes Phase 2 execution plans and generates robust Python code that:
    1. Sanitizes data before computation
    2. Uses correct aggregation strategy
    3. Returns standardized results with statistical validation
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize planner.
        
        Args:
            df: DataFrame for schema inspection
        """
        self.df = df
        self.templates = AggregationTemplates()
    
    def create_execution_plan(
        self, 
        phase2_plan: Phase2ExecutionPlan
    ) -> Phase3ExecutionPlan:
        """
        Create Phase 3 execution plan from Phase 2 plan.
        
        Args:
            phase2_plan: Execution plan from Phase 2
            
        Returns:
            Phase 3 execution plan with generated code
        """
        logger.info(f"Generating code for: {phase2_plan.hypothesis_title}")
        
        # Extract columns from polars_expression
        columns_used = self._extract_columns(phase2_plan.polars_expression)
        
        # Determine denominator column for zero filtering
        denominator_col = phase2_plan.denominator_column if hasattr(phase2_plan, 'denominator_column') else None
        
        # Determine aggregation scope
        aggregation_scope = phase2_plan.aggregation_scope if hasattr(phase2_plan, 'aggregation_scope') else "GLOBAL"
        
        # Get dimensions
        dimensions = phase2_plan.dimensions if hasattr(phase2_plan, 'dimensions') else []
        
        # Generate complete code using templates
        generated_code = self.templates.generate_complete_code(
            polars_expression=phase2_plan.polars_expression,
            aggregation_scope=aggregation_scope,
            dimensions=dimensions,
            time_column=None,  # TODO: Extract from hypothesis
            time_grain=None,   # TODO: Extract from hypothesis
            columns_used=columns_used,
            denominator_col=denominator_col,
            analysis_type="ratio"  # TODO: Determine from metric_template
        )
        
        logger.info(f"✓ Generated {len(generated_code)} chars of code")
        
        return Phase3ExecutionPlan(
            hypothesis_id=phase2_plan.hypothesis_id,
            hypothesis_title=phase2_plan.hypothesis_title,
            generated_code=generated_code,
            aggregation_scope=aggregation_scope,
            dimensions=dimensions,
            polars_expression=phase2_plan.polars_expression,
            columns_used=columns_used
        )
    
    def _extract_columns(self, polars_expression: str) -> List[str]:
        """
        Extract column names from polars expression.
        
        Args:
            polars_expression: Polars expression string
            
        Returns:
            List of column names
        """
        import re
        
        # Find all pl.col('column_name') patterns
        pattern = r"pl\.col\(['\"]([^'\"]+)['\"]\)"
        matches = re.findall(pattern, polars_expression)
        
        # Remove duplicates and return
        return list(set(matches))
