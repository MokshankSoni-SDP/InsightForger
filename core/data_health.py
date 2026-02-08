"""
Data Health Checker Module.

Validates data quality before analysis to prevent unreliable insights.
Checks for: null percentages, zero variance, high cardinality, and overall quality score.
"""
import polars as pl
from dataclasses import dataclass
from typing import List, Dict
from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class DataHealthReport:
    """Report on dataset quality."""
    overall_score: float  # 0-100
    warnings: List[str]
    critical_issues: List[str]
    column_scores: Dict[str, float]
    should_proceed: bool
    row_count: int
    column_count: int


class DataHealthChecker:
    """Validates data quality and generates health reports."""
    
    def __init__(self, df: pl.DataFrame, min_score: float = 70.0):
        self.df = df
        self.min_score = min_score
        self.row_count = len(df)
        self.column_count = len(df.columns)
    
    def check_health(self) -> DataHealthReport:
        """
        Perform comprehensive data health check.
        
        Returns:
            DataHealthReport with quality metrics and recommendations
        """
        logger.info("Starting data health check")
        
        warnings = []
        critical_issues = []
        column_scores = {}
        
        # Check 1: Null percentage per column
        self._check_null_percentages(column_scores, warnings, critical_issues)
        
        # Check 2: Zero variance (constant columns)
        self._check_zero_variance(column_scores, warnings)
        
        # Check 3: High cardinality explosion
        self._check_high_cardinality(warnings)
        
        # Check 4: Minimum row count
        if self.row_count < 30:
            critical_issues.append(f"Insufficient rows: {self.row_count} < 30 (statistical power too low)")
        
        # Calculate overall score
        if column_scores:
            overall_score = sum(column_scores.values()) / len(column_scores)
        else:
            overall_score = 0.0
        
        # Penalize for critical issues
        if critical_issues:
            overall_score *= 0.5
        
        should_proceed = (
            overall_score >= self.min_score and 
            len(critical_issues) == 0
        )
        
        logger.info(f"Data Health Score: {overall_score:.1f}/100")
        logger.info(f"Warnings: {len(warnings)}, Critical Issues: {len(critical_issues)}")
        
        return DataHealthReport(
            overall_score=overall_score,
            warnings=warnings,
            critical_issues=critical_issues,
            column_scores=column_scores,
            should_proceed=should_proceed,
            row_count=self.row_count,
            column_count=self.column_count
        )
    
    def _check_null_percentages(
        self, 
        column_scores: Dict[str, float], 
        warnings: List[str], 
        critical_issues: List[str]
    ):
        """Check null percentage for each column."""
        for col in self.df.columns:
            null_count = self.df[col].null_count()
            null_pct = (null_count / self.row_count) * 100 if self.row_count > 0 else 0
            
            if null_pct > 50:
                critical_issues.append(f"{col}: {null_pct:.1f}% nulls (> 50%)")
                column_scores[col] = 0.0
            elif null_pct > 20:
                warnings.append(f"{col}: {null_pct:.1f}% nulls (> 20%)")
                column_scores[col] = 50.0
            elif null_pct > 5:
                column_scores[col] = 80.0
            else:
                column_scores[col] = 100.0
    
    def _check_zero_variance(self, column_scores: Dict[str, float], warnings: List[str]):
        """Check for constant-value columns (zero variance)."""
        for col in self.df.columns:
            if self.df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
                try:
                    variance = self.df[col].var()
                    if variance is not None and variance == 0:
                        warnings.append(f"{col}: Zero variance (constant value)")
                        # Reduce score but don't make it critical
                        current_score = column_scores.get(col, 100.0)
                        column_scores[col] = min(current_score, 30.0)
                except:
                    # Skip if variance calculation fails
                    pass
    
    def _check_high_cardinality(self, warnings: List[str]):
        """Check for high cardinality columns that may cause explosion."""
        for col in self.df.columns:
            unique_count = self.df[col].n_unique()
            unique_ratio = unique_count / self.row_count if self.row_count > 0 else 0
            
            # High cardinality warning
            if unique_count > 1000:
                warnings.append(
                    f"{col}: {unique_count:,} unique values (high cardinality, may need Pareto filtering)"
                )
            
            # Potential identifier (very high uniqueness)
            if unique_ratio > 0.95 and unique_count > 100:
                warnings.append(
                    f"{col}: {unique_ratio:.1%} unique (likely identifier, limited analytical value)"
                )
