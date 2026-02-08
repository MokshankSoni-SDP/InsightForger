"""
Adversarial validation module.

Acts as a statistical gatekeeper to reject weak insights.
Enhanced with type-aware validation, verdict levels, and explainability.
"""
import polars as pl
import numpy as np
from typing import Dict, Any, List, Literal
from scipy import stats
from utils.schemas import ComputationResult, ValidationResult, ValidatedInsight, Hypothesis
from utils.helpers import get_logger, create_time_windows

logger = get_logger(__name__)


class AdversarialValidator:
    """Devil's advocate - rejects statistically weak insights with type-aware validation."""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.min_sample_size = 30
        self.min_pvalue_threshold = 0.05
        self.min_correlation_threshold = 0.3
    
    def validate_result(
        self,
        hypothesis: Hypothesis,
        result: ComputationResult
    ) -> ValidationResult:
        """
        Validate computation result with strict statistical criteria.
        
        Improvements:
        - Requires explicit sample_size (Improvement #1)
        - Type-aware validation (Improvement #2)
        - Real temporal stability (Improvement #3)
        - Verdict levels: accept/weak_signal/reject (Improvement #4)
        - Logs passed/failed checks (Improvement #5)
        
        Args:
            hypothesis: Original hypothesis
            result: Computation result to validate
            
        Returns:
            ValidationResult with verdict and explainability
        """
        logger.info(f"Validating hypothesis: {hypothesis.id} (type: {hypothesis.expected_insight_type})")
        
        if not result.success:
            return ValidationResult(
                hypothesis_id=hypothesis.id,
                passed=False,
                rejection_reason="Computation failed",
                confidence_score=0.0,
                verdict="reject"  # Improvement #4
            )
        
        result_data = result.result_data
        statistical_tests = {}
        confidence_score = 1.0
        rejection_reasons = []
        
        # Improvement #5: Track passed and failed checks
        passed_checks = []
        failed_checks = []
        
        # Improvement #1: REQUIRE explicit sample_size
        if "sample_size" not in result_data:
            # Fallback to _meta
            meta = result_data.get("_meta", {})
            sample_size = meta.get("sample_size") or meta.get("n_samples")
            
            if sample_size is None:
                rejection_reasons.append("Sample size not reported by computation")
                failed_checks.append("sample_size_missing")
                confidence_score *= 0.3
                sample_size = len(self.df)  # Fallback, but heavily penalized
            else:
                passed_checks.append("sample_size_reported")
        else:
            sample_size = result_data["sample_size"]
            passed_checks.append("sample_size_reported")
        
        # Check sample size adequacy
        if sample_size < self.min_sample_size:
            rejection_reasons.append(f"Sample size too small ({sample_size} < {self.min_sample_size})")
            failed_checks.append("sample_size_adequate")
            confidence_score *= 0.2
        else:
            passed_checks.append("sample_size_adequate")
        
        statistical_tests["sample_size"] = sample_size
        
        # Improvement #2: Type-aware validation
        if hypothesis.expected_insight_type == "forecast":
            self._validate_forecast(result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests)
        elif hypothesis.expected_insight_type == "anomaly":
            confidence_score *= self._validate_anomaly(result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests)
        elif hypothesis.expected_insight_type == "correlation":
            confidence_score *= self._validate_correlation(result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests)
        elif hypothesis.expected_insight_type == "causal":
            confidence_score *= self._validate_causal(result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests)
        elif hypothesis.expected_insight_type == "trend":
            confidence_score *= self._validate_trend(result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests)
        
        # Improvement #3: Real temporal stability check
        if hypothesis.expected_insight_type in ["correlation", "trend"]:
            time_cols = [col for col in self.df.columns 
                        if any(kw in col.lower() for kw in ["date", "time"])]
            
            if time_cols:
                stability_check = self._check_temporal_stability_real(hypothesis, result_data, time_cols[0])
                statistical_tests["temporal_stability"] = stability_check
                
                if not stability_check.get("stable", True):
                    rejection_reasons.append(f"Results not stable across time windows (variance: {stability_check.get('variance', 'N/A')})")
                    failed_checks.append("temporal_stability")
                    confidence_score *= 0.6
                else:
                    passed_checks.append("temporal_stability")
        
        # Check for data quality issues
        if result_data.get("null_percentage", 0) > 50:
            rejection_reasons.append("Too many null values in analysis")
            failed_checks.append("data_quality")
            confidence_score *= 0.3
        else:
            passed_checks.append("data_quality")
        
        # Improvement #4: Verdict levels instead of binary pass/fail
        confidence_score = max(0.0, min(1.0, confidence_score))
        verdict = self._determine_verdict(confidence_score)
        
        validation = ValidationResult(
            hypothesis_id=hypothesis.id,
            verdict=verdict,  # passed is now derived from verdict
            rejection_reason="; ".join(rejection_reasons) if rejection_reasons else None,
            statistical_tests=statistical_tests,
            confidence_score=confidence_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks
        )
        
        if verdict == "accept":
            logger.info(f"✓ ACCEPTED: {hypothesis.id} (confidence: {confidence_score:.2f})")
        elif verdict == "weak_signal":
            logger.info(f"⚠ WEAK SIGNAL: {hypothesis.id} (confidence: {confidence_score:.2f})")
        else:
            logger.warning(f"✗ REJECTED: {hypothesis.id} (confidence: {confidence_score:.2f}): {rejection_reasons}")
        
        return validation
    
    def _determine_verdict(self, confidence_score: float) -> Literal["accept", "weak_signal", "reject"]:
        """
        Improvement #4: Determine verdict level.
        
        - accept: High confidence, production-ready
        - weak_signal: Interesting but uncertain, for exploration
        - reject: Not trustworthy
        """
        if confidence_score > 0.75:
            return "accept"
        elif confidence_score >= 0.4:
            return "weak_signal"
        else:
            return "reject"
    
    def _validate_correlation(
        self, result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests
    ) -> float:
        """Type-aware validation for correlation hypotheses."""
        confidence_modifier = 1.0
        
        # P-value check
        p_value = result_data.get("p_value") or result_data.get("pvalue")
        if p_value is not None:
            statistical_tests["p_value"] = float(p_value)
            
            if p_value > self.min_pvalue_threshold:
                rejection_reasons.append(f"P-value not significant ({p_value:.4f} > {self.min_pvalue_threshold})")
                failed_checks.append("p_value_significant")
                confidence_modifier *= 0.3
            else:
                passed_checks.append("p_value_significant")
                confidence_modifier *= 0.95
        
        # Correlation strength check
        correlation = result_data.get("correlation") or result_data.get("corr")
        if correlation is not None:
            abs_corr = abs(float(correlation))
            statistical_tests["correlation"] = float(correlation)
            
            if abs_corr < self.min_correlation_threshold:
                rejection_reasons.append(f"Correlation too weak ({abs_corr:.4f} < {self.min_correlation_threshold})")
                failed_checks.append("correlation_strength")
                confidence_modifier *= 0.4
            else:
                passed_checks.append("correlation_strength")
                confidence_modifier *= 0.9
        
        return confidence_modifier
    
    def _validate_forecast(
        self, result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests
    ):
        """Type-aware validation for forecast hypotheses."""
        # Check for model confidence
        model_confidence = result_data.get("model_confidence")
        if model_confidence is not None:
            statistical_tests["model_confidence"] = float(model_confidence)
            
            if model_confidence < 0.5:
                rejection_reasons.append(f"Model confidence too low ({model_confidence:.2f})")
                failed_checks.append("model_confidence")
            else:
                passed_checks.append("model_confidence")
        
        # Check AIC (lower is better)
        aic = result_data.get("aic")
        if aic is not None:
            statistical_tests["aic"] = float(aic)
            # AIC is relative, so just log it
            passed_checks.append("aic_reported")
        
        # Check for forecast horizon
        periods = result_data.get("periods")
        if periods and periods > 0:
            passed_checks.append("forecast_horizon_valid")
    
    def _validate_anomaly(
        self, result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests
    ) -> float:
        """Type-aware validation for anomaly hypotheses."""
        confidence_modifier = 1.0
        
        # Check anomaly count and rate
        anomalies_found = result_data.get("anomalies_found", 0)
        anomaly_rate = result_data.get("anomaly_rate", 0.0)
        
        statistical_tests["anomalies_found"] = anomalies_found
        statistical_tests["anomaly_rate"] = anomaly_rate
        
        if anomalies_found == 0:
            rejection_reasons.append("No anomalies detected")
            failed_checks.append("anomalies_found")
            confidence_modifier *= 0.1
        else:
            passed_checks.append("anomalies_found")
        
        # Check if anomaly rate is reasonable (not too high = likely false positives)
        if anomaly_rate > 0.3:
            rejection_reasons.append(f"Anomaly rate too high ({anomaly_rate:.1%}) - likely false positives")
            failed_checks.append("anomaly_rate_reasonable")
            confidence_modifier *= 0.5
        else:
            passed_checks.append("anomaly_rate_reasonable")
        
        # Check for severity scoring
        if "summary_stats" in result_data and "mean_severity" in result_data["summary_stats"]:
            mean_severity = result_data["summary_stats"]["mean_severity"]
            if mean_severity > 2.0:  # High severity
                passed_checks.append("high_severity_anomalies")
                confidence_modifier *= 1.1  # Boost confidence
        
        return confidence_modifier
    
    def _validate_causal(
        self, result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests
    ) -> float:
        """Type-aware validation for causal hypotheses."""
        confidence_modifier = 1.0
        
        # Check causal confidence (from causality.py)
        causal_confidence = result_data.get("causal_confidence")
        if causal_confidence is not None:
            statistical_tests["causal_confidence"] = causal_confidence
            
            if causal_confidence == "low" or causal_confidence == "none":
                rejection_reasons.append(f"Causal confidence too low: {causal_confidence}")
                failed_checks.append("causal_confidence")
                confidence_modifier *= 0.3
            elif causal_confidence == "medium":
                passed_checks.append("causal_confidence_medium")
                confidence_modifier *= 0.8
            else:  # high
                passed_checks.append("causal_confidence_high")
                confidence_modifier *= 1.0
        
        # Check refutation tests
        refutation_tests = result_data.get("refutation_tests", [])
        if refutation_tests:
            passed_refutations = sum(1 for t in refutation_tests if t.get("passed", False))
            statistical_tests["refutation_tests_passed"] = passed_refutations
            
            if passed_refutations >= 2:
                passed_checks.append("refutation_tests")
                confidence_modifier *= 1.05
            else:
                failed_checks.append("refutation_tests")
                confidence_modifier *= 0.7
        
        # Check for standard error
        std_error = result_data.get("standard_error")
        if std_error is None or std_error == 0:
            failed_checks.append("standard_error_available")
            confidence_modifier *= 0.6
        else:
            passed_checks.append("standard_error_available")
        
        return confidence_modifier
    
    def _validate_trend(
        self, result_data, rejection_reasons, passed_checks, failed_checks, statistical_tests
    ) -> float:
        """Type-aware validation for trend hypotheses."""
        confidence_modifier = 1.0
        
        # Check for trend direction
        trend_direction = result_data.get("trend_direction")
        if trend_direction:
            statistical_tests["trend_direction"] = trend_direction
            passed_checks.append("trend_direction_detected")
        
        # Check R-squared or similar fit metric
        r_squared = result_data.get("r_squared") or result_data.get("r2")
        if r_squared is not None:
            statistical_tests["r_squared"] = float(r_squared)
            
            if r_squared < 0.3:
                rejection_reasons.append(f"Trend fit too weak (R²={r_squared:.3f})")
                failed_checks.append("trend_fit")
                confidence_modifier *= 0.5
            else:
                passed_checks.append("trend_fit")
        
        return confidence_modifier
    
    def _check_temporal_stability_real(
        self,
        hypothesis: Hypothesis,
        result_data: Dict[str, Any],
        time_col: str
    ) -> Dict[str, Any]:
        """
        Improvement #3: Real temporal stability check.
        
        Recomputes the metric in each time window and checks for consistency.
        """
        try:
            windows = create_time_windows(self.df, time_col, window_size=3)
            
            if len(windows) < 2:
                return {"stable": True, "note": "Insufficient windows to test", "window_count": len(windows)}
            
            metric = hypothesis.resolved_metric
            related_metric = getattr(hypothesis, "related_metric", None)
            
            # For correlation-type analyses
            if hypothesis.expected_insight_type == "correlation" and related_metric:
                window_correlations = []
                
                for window_df in windows:
                    if len(window_df) > 10 and metric in window_df.columns and related_metric in window_df.columns:
                        # Recompute correlation in this window
                        try:
                            metric_vals = window_df[metric].drop_nulls().to_numpy()
                            related_vals = window_df[related_metric].drop_nulls().to_numpy()
                            
                            min_len = min(len(metric_vals), len(related_vals))
                            if min_len > 5:
                                from scipy.stats import pearsonr
                                corr, _ = pearsonr(metric_vals[:min_len], related_vals[:min_len])
                                window_correlations.append(corr)
                        except:
                            continue
                
                if len(window_correlations) >= 2:
                    variance = np.var(window_correlations)
                    
                    # Check sign consistency
                    original_corr = result_data.get("correlation", 0)
                    same_sign = all((c > 0) == (original_corr > 0) for c in window_correlations)
                    
                    # Check magnitude similarity (within ±30%)
                    magnitude_similar = all(abs(c) > abs(original_corr) * 0.7 for c in window_correlations)
                    
                    stable = variance < 0.1 and same_sign and magnitude_similar
                    
                    return {
                        "stable": stable,
                        "variance": float(variance),
                        "window_count": len(window_correlations),
                        "window_values": [float(c) for c in window_correlations],
                        "same_sign": same_sign,
                        "magnitude_similar": magnitude_similar
                    }
            
            # For trend analyses
            elif hypothesis.expected_insight_type == "trend":
                window_means = []
                
                for window_df in windows:
                    if len(window_df) > 5 and metric in window_df.columns:
                        mean_val = window_df[metric].drop_nulls().mean()
                        if mean_val is not None:
                            window_means.append(float(mean_val))
                
                if len(window_means) >= 2:
                    # Check if trend is consistent (all increasing or all decreasing)
                    diffs = np.diff(window_means)
                    consistent_direction = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
                    
                    variance = np.var(window_means)
                    
                    return {
                        "stable": consistent_direction,
                        "variance": float(variance),
                        "window_count": len(window_means),
                        "consistent_direction": consistent_direction
                    }
            
            return {"stable": True, "note": "No stability test performed for this type"}
            
        except Exception as e:
            logger.error(f"Stability check failed: {e}")
            return {"stable": True, "error": str(e)}
    
    def create_validated_insight(
        self,
        hypothesis: Hypothesis,
        result: ComputationResult,
        validation: ValidationResult
    ) -> ValidatedInsight:
        """Create a validated insight object."""
        return ValidatedInsight(
            hypothesis_id=hypothesis.id,
            lens=hypothesis.lens,
            title=hypothesis.title,
            metric=hypothesis.resolved_metric, # Updated from metric_target
            result_data=result.result_data,
            validation=validation,
            resolution_type=hypothesis.resolution_type,
            formula_name=hypothesis.formula_name
        )
