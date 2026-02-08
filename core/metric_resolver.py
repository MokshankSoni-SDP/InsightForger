"""
Phase 2: Autonomous Metric Resolver with Polars Formula Factory
"""
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import polars as pl
from utils.schemas import (
    BusinessContext, SemanticProfile, Hypothesis,
    ExecutionPlan, AggregationScope, EntityProfile
)
from utils.helpers import get_logger

load_dotenv()
logger = get_logger(__name__)


# ============================================================================
# POLARS FORMULA FACTORY
# ============================================================================

def _format_dims(dims):
    """Helper to format dimensions for f-string."""
    return ', '.join([f"'{d}'" for d in dims])


class PolarsFormulaBuilder:
    """
    Dynamic formula generator for Polars expressions.
    
    Builds executable Polars code based on mathematical relationships,
    not hardcoded column names.
    """
    
    # Formula templates (lambda functions for dynamic generation)
    FORMULA_TEMPLATES = {
        "SAME_ROW": lambda n, d: f"pl.col('{n}') / pl.col('{d}')",
        "GLOBAL_SUM": lambda n, d: f"pl.col('{n}') / pl.col('{d}').sum()",
        "GROUP_SUM": lambda n, d, dims: f"pl.col('{n}').sum() / pl.col('{d}').sum().over([{_format_dims(dims)}])",
        "DIMENSIONAL": lambda n, d: f"pl.col('{n}').sum() / pl.col('{d}').sum()",
        "MEAN": lambda n: f"pl.col('{n}').mean()",
        "SUM": lambda n: f"pl.col('{n}').sum()",
        "GROWTH": lambda n: f"(pl.col('{n}') - pl.col('{n}').shift(1)) / pl.col('{n}').shift(1) * 100"
    }
    
    def build_ratio_formula(
        self,
        numerator: str,
        denominator: Optional[str],
        scope: Optional[str],
        dimensions: List[str] = None
    ) -> str:
        """
        Build Polars expression for ratio calculation.
        
        Args:
            numerator: Physical column name for numerator
            denominator: Physical column name for denominator (None for non-ratio metrics)
            scope: Denominator scope (SAME_ROW, GLOBAL_SUM, GROUP_SUM, DIMENSIONAL)
            dimensions: List of dimension columns for grouping
        
        Returns:
            Executable Polars expression string
        """
        dimensions = dimensions or []
        
        # Non-ratio metrics (MEAN, SUM, GROWTH)
        if not denominator:
            return self.FORMULA_TEMPLATES["MEAN"](numerator)
        
        # Ratio metrics
        if scope == "SAME_ROW":
            formula = self.FORMULA_TEMPLATES["SAME_ROW"](numerator, denominator)
        elif scope == "GLOBAL_SUM":
            formula = self.FORMULA_TEMPLATES["GLOBAL_SUM"](numerator, denominator)
        elif scope == "GROUP_SUM" and dimensions:
            formula = self.FORMULA_TEMPLATES["GROUP_SUM"](numerator, denominator, dimensions)
        elif scope == "DIMENSIONAL":
            formula = self.FORMULA_TEMPLATES["DIMENSIONAL"](numerator, denominator)
        else:
            # Default to SAME_ROW if scope not recognized
            logger.warning(f"Unknown scope '{scope}', defaulting to SAME_ROW")
            formula = self.FORMULA_TEMPLATES["SAME_ROW"](numerator, denominator)
        
        # Add zero-division protection
        formula = self._add_zero_protection(formula, denominator)
        
        return formula
    
    def _add_zero_protection(self, formula: str, denominator: str) -> str:
        """Add null-if-zero protection to prevent division by zero."""
        # Wrap denominator in replace(0, None) to convert zeros to nulls
        protected_den = f"pl.col('{denominator}').replace(0, None)"
        formula = formula.replace(f"pl.col('{denominator}')", protected_den)
        return formula
    
    def apply_guardrail(self, col_name: str, guardrail_transformation: str) -> str:
        """
        Apply guardrail transformation to column.
        
        Args:
            col_name: Physical column name
            guardrail_transformation: Transformation formula (e.g., "INVERT(our_position) = (16 - our_position)")
        
        Returns:
            Transformed column expression
        """
        if not guardrail_transformation:
            return f"pl.col('{col_name}')"
        
        # Parse transformation
        if "INVERT" in guardrail_transformation:
            # Extract formula: "INVERT(our_position) = (16 - our_position)"
            match = re.search(r'=\s*\(([^)]+)\)', guardrail_transformation)
            if match:
                transform = match.group(1)
                # Replace column name with pl.col() syntax
                transform = transform.replace(col_name, f"pl.col('{col_name}')")
                return f"({transform})"
        
        elif "FILTER" in guardrail_transformation:
            # Data quality filter - apply as filter, not transformation
            return f"pl.col('{col_name}')"
        
        # Default: no transformation
        return f"pl.col('{col_name}')"


# ============================================================================
# AUTONOMOUS METRIC RESOLVER
# ============================================================================

class MetricResolver:
    """
    100% Autonomous Metric Resolver.
    
    Converts business concepts to executable Polars expressions using:
    - Semantic column mapping (not hardcoded synonyms)
    - Dynamic formula generation
    - Type safety validation
    - Guardrail injection
    """
    
    def __init__(self, profile: SemanticProfile, df: pl.DataFrame):
        """
        Initialize resolver with semantic profile and dataset.
        
        Args:
            profile: Semantic profile from Phase 0.3
            df: Polars DataFrame to execute against
        """
        self.profile = profile
        self.df = df
        self.formula_builder = PolarsFormulaBuilder()
    
    def resolve_hypothesis(self, hypothesis: Hypothesis) -> ExecutionPlan:
        """
        Main entry point: Convert hypothesis to execution plan.
        
        Steps:
        1. Find real columns for numerator/denominator
        2. Validate types
        3. Build Polars formula
        4. Apply guardrails
        5. Return execution plan
        
        Args:
            hypothesis: Hypothesis from Phase 1
        
        Returns:
            ExecutionPlan with validated formula
        
        Raises:
            ValueError: If column mapping or validation fails
        """
        logger.info(f"Resolving hypothesis: {hypothesis.title}")
        
        # Step 1: Column mapping
        num_col, num_method, num_confidence = self._find_column_by_concept(
            hypothesis.numerator_concept
        )
        
        if not num_col:
            raise ValueError(f"Could not find column for numerator concept: {hypothesis.numerator_concept}")
        
        den_col = None
        den_method = "none"
        den_confidence = 1.0
        
        if hypothesis.denominator_concept:
            den_col, den_method, den_confidence = self._find_column_by_concept(
                hypothesis.denominator_concept
            )
            if not den_col:
                raise ValueError(f"Could not find column for denominator concept: {hypothesis.denominator_concept}")
        
        # Step 2: Type validation
        if not self._validate_column_types(num_col, den_col):
            raise ValueError(f"Type validation failed for {hypothesis.id}")
        
        # Step 3: Apply guardrails to columns
        num_expr = num_col
        den_expr = den_col
        
        if hypothesis.guardrail_transformation:
            # Check if guardrail applies to numerator or denominator
            if hypothesis.guardrail_applied and hypothesis.guardrail_applied in num_col:
                num_expr = self.formula_builder.apply_guardrail(num_col, hypothesis.guardrail_transformation)
            elif hypothesis.guardrail_applied and den_col and hypothesis.guardrail_applied in den_col:
                den_expr = self.formula_builder.apply_guardrail(den_col, hypothesis.guardrail_transformation)
        
        # Step 4: Build formula
        formula = self.formula_builder.build_ratio_formula(
            numerator=num_col,
            denominator=den_col,
            scope=hypothesis.denominator_scope,
            dimensions=hypothesis.dimensions
        )
        
        # Step 5: Create execution plan
        resolution_confidence = min(num_confidence, den_confidence) if den_col else num_confidence
        
        return ExecutionPlan(
            hypothesis_id=hypothesis.id,
            numerator_column=num_col,
            denominator_column=den_col,
            polars_expression=formula,
            aggregation_scope=hypothesis.aggregation_scope,
            time_grain=hypothesis.time_grain,
            dimensions=hypothesis.dimensions,
            type_safe=True,
            has_guardrails=bool(hypothesis.guardrail_transformation),
            null_safe=True,  # Zero-division protection added
            resolution_method=num_method,
            resolution_confidence=resolution_confidence,
            hypothesis_title=hypothesis.title,
            lens_name=hypothesis.lens
        )
    
    def _find_column_by_concept(self, concept: str) -> Tuple[Optional[str], str, float]:
        """
        Find physical column for business concept using semantic matching.
        
        Priority:
        1. Exact name match (case-insensitive)
        2. Semantic role match
        3. Fuzzy/similarity match
        
        Args:
            concept: Business concept (e.g., "revenue", "sales", "efficiency")
        
        Returns:
            Tuple of (column_name, method, confidence)
        """
        concept_lower = concept.lower().strip()
        
        # Priority 1: Exact match
        for col in self.df.columns:
            if col.lower() == concept_lower:
                logger.info(f"Exact match: '{concept}' → '{col}'")
                return col, "exact", 1.0
        
        # Priority 2: Semantic match
        semantic_match = self._semantic_match(concept)
        if semantic_match:
            col, confidence = semantic_match
            logger.info(f"Semantic match: '{concept}' → '{col}' (confidence: {confidence:.2f})")
            return col, "semantic", confidence
        
        # Priority 3: Fuzzy match
        fuzzy_match = self._fuzzy_match(concept)
        if fuzzy_match:
            col, confidence = fuzzy_match
            logger.info(f"Fuzzy match: '{concept}' → '{col}' (confidence: {confidence:.2f})")
            return col, "fuzzy", confidence
        
        logger.error(f"No match found for concept: '{concept}'")
        return None, "failed", 0.0
    
    def _semantic_match(self, concept: str) -> Optional[Tuple[str, float]]:
        """
        Find column by semantic role.
        
        Example: "revenue" matches column with semantic_role="financial"
        """
        concept_lower = concept.lower()
        
        # Semantic role keywords
        role_keywords = {
            "financial": ["revenue", "sales", "income", "profit", "cost", "price", "margin"],
            "funnel": ["clicks", "impressions", "conversions", "leads", "visits"],
            "temporal": ["date", "time", "timestamp", "period", "month", "week"],
            "identifier": ["id", "key", "code", "number"],
            "categorical": ["category", "type", "class", "group", "segment"]
        }
        
        # Find which role this concept belongs to
        target_role = None
        for role, keywords in role_keywords.items():
            if any(kw in concept_lower for kw in keywords):
                target_role = role
                break
        
        if not target_role:
            return None
        
        # Search for columns with matching semantic role
        for entity in self.profile.entities:
            if hasattr(entity, 'columns'):
                for col_obj in entity.columns:
                    if col_obj.semantic_role == target_role:
                        # Check if column exists in dataframe
                        if col_obj.name in self.df.columns:
                            return col_obj.name, 0.8
        
        return None
    
    def _fuzzy_match(self, concept: str) -> Optional[Tuple[str, float]]:
        """
        Find column by string similarity (last resort).
        
        Uses simple substring matching.
        """
        concept_lower = concept.lower()
        best_match = None
        best_score = 0.0
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Substring match
            if concept_lower in col_lower or col_lower in concept_lower:
                score = len(concept_lower) / max(len(concept_lower), len(col_lower))
                if score > best_score:
                    best_score = score
                    best_match = col
        
        if best_match and best_score >= 0.5:
            return best_match, best_score
        
        return None
    
    def _validate_column_types(self, num_col: str, den_col: Optional[str]) -> bool:
        """
        Validate columns are numeric and suitable for math.
        
        Args:
            num_col: Numerator column name
            den_col: Denominator column name (optional)
        
        Returns:
            True if valid, False otherwise
        """
        # Check numerator is numeric
        if not self.df[num_col].dtype.is_numeric():
            logger.error(f"Numerator '{num_col}' is not numeric: {self.df[num_col].dtype}")
            return False
        
        # Check denominator is numeric (if exists)
        if den_col and not self.df[den_col].dtype.is_numeric():
            logger.error(f"Denominator '{den_col}' is not numeric: {self.df[den_col].dtype}")
            return False
        
        # Check null percentage
        null_pct = self.df[num_col].null_count() / len(self.df)
        if null_pct > 0.7:
            logger.warning(f"Column '{num_col}' is {null_pct*100:.1f}% null (high)")
            return False
        
        logger.info(f"Type validation passed: {num_col}" + (f", {den_col}" if den_col else ""))
        return True
