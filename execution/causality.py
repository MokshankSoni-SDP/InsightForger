"""
Causality analysis module using DoWhy.

Implements strict causal inference with explicit confounder specification,
refutation tests, and causal confidence labeling.
Enhanced to prevent misleading causal claims.
"""
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from utils.helpers import get_logger

logger = get_logger(__name__)


class CausalityEngine:
    """Causal analysis using DoWhy framework with strict validation."""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        # Convert to pandas for DoWhy compatibility
        self.df_pandas = df.to_pandas()
    
    def estimate_causal_effect_strict(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str],  # Improvement #1: MANDATORY, not Optional
        instrument: Optional[str] = None,
        run_refutation: bool = True
    ) -> Dict[str, Any]:
        """
        Improvement #5: Strict causal estimation with explicit confounders.
        
        This method REQUIRES explicit confounder specification and runs
        refutation tests by default. Use this for rigorous causal claims.
        
        Args:
            treatment: Treatment variable (cause)
            outcome: Outcome variable (effect)
            confounders: EXPLICIT list of confounding variables (MANDATORY)
            instrument: Optional instrumental variable
            run_refutation: Whether to run refutation tests (default: True)
            
        Returns:
            Dictionary with causal estimates, confidence, and validation results
        """
        logger.info(f"Estimating causal effect (STRICT): {treatment} → {outcome}")
        logger.info(f"Confounders: {confounders}")
        
        # Improvement #1: Reject if confounders not provided
        if not confounders or len(confounders) == 0:
            logger.error("Confounders not explicitly defined - rejecting causal estimation")
            return {
                "status": "rejected",
                "reason": "Confounders must be explicitly specified for causal inference",
                "recommendation": "Use estimate_association() for exploratory analysis",
                "causal_confidence": "none"
            }
        
        try:
            from dowhy import CausalModel
            
            # Validate confounders exist in data
            missing_vars = [c for c in confounders if c not in self.df.columns]
            if missing_vars:
                return {
                    "status": "rejected",
                    "reason": f"Confounders not found in data: {missing_vars}",
                    "causal_confidence": "none"
                }
            
            # Improvement #2: Build causal graph with explicit structure
            # Don't assume all confounders affect both treatment and outcome
            graph = self._build_causal_graph(treatment, outcome, confounders)
            
            # Create causal model
            model = CausalModel(
                data=self.df_pandas,
                treatment=treatment,
                outcome=outcome,
                graph=graph
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect()
            
            # Estimate effect with linear regression (default)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            # Calculate confidence intervals
            std_error = estimate.get_standard_error() if hasattr(estimate, 'get_standard_error') else None
            
            if std_error is None or std_error == 0:
                logger.warning("Standard error unavailable - low confidence estimate")
                causal_confidence = "low"
            else:
                causal_confidence = "medium"
            
            # Improvement #3: Run refutation tests
            refutation_results = []
            if run_refutation:
                logger.info("Running refutation tests...")
                refutation_results = self._run_refutation_tests(
                    model, identified_estimand, estimate
                )
                
                # Downgrade confidence if refutation fails
                if any(not r['passed'] for r in refutation_results):
                    logger.warning("Refutation tests failed - downgrading confidence")
                    if causal_confidence == "medium":
                        causal_confidence = "low"
            
            # Improvement #2: Assess overall causal confidence
            final_confidence, assumptions, warnings = self._assess_causal_confidence(
                estimate, std_error, confounders, refutation_results
            )
            
            result = {
                "status": "success",
                "treatment": treatment,
                "outcome": outcome,
                "estimated_effect": float(estimate.value),
                "standard_error": float(std_error) if std_error else None,
                "confidence_interval": {
                    "lower": float(estimate.value - 1.96 * std_error) if std_error else None,
                    "upper": float(estimate.value + 1.96 * std_error) if std_error else None
                },
                "confounders": confounders,
                "method": "backdoor.linear_regression",
                
                # Improvement #2: Explicit causal confidence and metadata
                "causal_confidence": final_confidence,  # "high", "medium", "low"
                "assumptions": assumptions,
                "warnings": warnings,
                "refutation_tests": refutation_results if run_refutation else []
            }
            
            logger.info(f"Causal effect: {estimate.value:.4f} (confidence: {final_confidence})")
            return result
            
        except Exception as e:
            logger.error(f"Causal estimation failed: {e}")
            # Improvement #4: NO correlation fallback - fail explicitly
            return {
                "status": "failed",
                "reason": str(e),
                "recommendation": "Use estimate_association() instead",
                "causal_confidence": "none"
            }
    
    def estimate_association(
        self,
        var1: str,
        var2: str
    ) -> Dict[str, Any]:
        """
        Improvement #5: Separate associational API.
        
        This method estimates ASSOCIATION, not causation.
        Use for exploratory analysis when confounders are unknown.
        
        Args:
            var1: First variable
            var2: Second variable
            
        Returns:
            Dictionary with correlation results (NOT causal)
        """
        logger.info(f"Estimating ASSOCIATION (not causal): {var1} ↔ {var2}")
        
        try:
            from scipy.stats import pearsonr, spearmanr
            
            var1_vals = self.df[var1].drop_nulls().to_numpy()
            var2_vals = self.df[var2].drop_nulls().to_numpy()
            
            # Ensure same length
            min_len = min(len(var1_vals), len(var2_vals))
            var1_vals = var1_vals[:min_len]
            var2_vals = var2_vals[:min_len]
            
            # Calculate both Pearson and Spearman
            pearson_corr, pearson_pval = pearsonr(var1_vals, var2_vals)
            spearman_corr, spearman_pval = spearmanr(var1_vals, var2_vals)
            
            return {
                "status": "success",
                "analysis_type": "association",  # NOT causation
                "var1": var1,
                "var2": var2,
                "pearson_correlation": float(pearson_corr),
                "pearson_pvalue": float(pearson_pval),
                "spearman_correlation": float(spearman_corr),
                "spearman_pvalue": float(spearman_pval),
                "n_samples": min_len,
                "interpretation": self._interpret_association(pearson_corr, pearson_pval),
                
                # Clear disclaimer
                "causal_claim": False,
                "warning": "This is correlation, NOT causation. Do not make causal claims from this result."
            }
            
        except Exception as e:
            logger.error(f"Association estimation failed: {e}")
            return {
                "status": "failed",
                "reason": str(e)
            }
    
    def _build_causal_graph(
        self, treatment: str, outcome: str, confounders: List[str]
    ) -> str:
        """
        Improvement #2: Build more conservative causal graph.
        
        Assumes confounders affect both treatment and outcome (backdoor).
        """
        # Build backdoor graph
        graph_edges = [f"{treatment} -> {outcome};"]
        
        for c in confounders:
            graph_edges.append(f"{c} -> {treatment};")
            graph_edges.append(f"{c} -> {outcome};")
        
        graph = f"""
        digraph {{
            {' '.join(graph_edges)}
        }}
        """
        
        return graph
    
    def _run_refutation_tests(
        self, model, identified_estimand, estimate
    ) -> List[Dict[str, Any]]:
        """
        Improvement #3: Run DoWhy refutation tests.
        
        Tests:
        1. Placebo treatment (should be ~0)
        2. Random common cause (should be stable)
        3. Data subset (should be stable)
        """
        refutation_results = []
        
        # Test 1: Placebo treatment refuter
        try:
            refute_placebo = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter"
            )
            
            # Check if new estimate is significantly different from 0
            new_effect = refute_placebo.new_effect if hasattr(refute_placebo, 'new_effect') else 0
            
            refutation_results.append({
                "test": "placebo_treatment",
                "passed": abs(new_effect) < 0.1 * abs(estimate.value) if estimate.value != 0 else abs(new_effect) < 0.01,
                "new_effect": float(new_effect) if new_effect else None,
                "interpretation": "Placebo should have ~0 effect"
            })
        except Exception as e:
            logger.warning(f"Placebo refutation failed: {e}")
            refutation_results.append({
                "test": "placebo_treatment",
                "passed": False,
                "error": str(e)
            })
        
        # Test 2: Random common cause refuter
        try:
            refute_random = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause"
            )
            
            new_effect = refute_random.new_effect if hasattr(refute_random, 'new_effect') else estimate.value
            
            # Effect should be similar after adding random confounder
            effect_change = abs(new_effect - estimate.value) / abs(estimate.value) if estimate.value != 0 else 0
            
            refutation_results.append({
                "test": "random_common_cause",
                "passed": effect_change < 0.2,  # Less than 20% change
                "new_effect": float(new_effect) if new_effect else None,
                "interpretation": "Effect should be stable after adding random confounder"
            })
        except Exception as e:
            logger.warning(f"Random common cause refutation failed: {e}")
            refutation_results.append({
                "test": "random_common_cause",
                "passed": False,
                "error": str(e)
            })
        
        # Test 3: Data subset refuter
        try:
            refute_subset = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8
            )
            
            new_effect = refute_subset.new_effect if hasattr(refute_subset, 'new_effect') else estimate.value
            effect_change = abs(new_effect - estimate.value) / abs(estimate.value) if estimate.value != 0 else 0
            
            refutation_results.append({
                "test": "data_subset",
                "passed": effect_change < 0.3,  # Less than 30% change
                "new_effect": float(new_effect) if new_effect else None,
                "interpretation": "Effect should be stable across data subsets"
            })
        except Exception as e:
            logger.warning(f"Data subset refutation failed: {e}")
            refutation_results.append({
                "test": "data_subset",
                "passed": False,
                "error": str(e)
            })
        
        return refutation_results
    
    def _assess_causal_confidence(
        self, estimate, std_error, confounders, refutation_results
    ) -> tuple[str, List[str], List[str]]:
        """
        Improvement #2: Assess overall causal confidence.
        
        Returns:
            (confidence_level, assumptions, warnings)
        """
        confidence = "medium"  # Start with medium
        assumptions = [
            "Linear relationship between treatment and outcome",
            "No unmeasured confounders beyond those specified",
            "Correct causal graph structure",
            "Positivity (overlap in treatment assignment)"
        ]
        warnings = []
        
        # Check standard error
        if std_error is None or std_error == 0:
            confidence = "low"
            warnings.append("Standard error unavailable or zero")
        
        # Check sample size via confounders
        n_confounders = len(confounders)
        if n_confounders > 10:
            confidence = "low"
            warnings.append(f"Many confounders ({n_confounders}) may lead to overfitting")
        
        # Check refutation results
        if refutation_results:
            failed_tests = [r for r in refutation_results if not r.get('passed', False)]
            if len(failed_tests) >= 2:
                confidence = "low"
                warnings.append(f"{len(failed_tests)}/3 refutation tests failed")
            elif len(failed_tests) == 1:
                if confidence == "medium":
                    pass  # Keep medium
                warnings.append("1 refutation test failed")
        
        # Upgrade to high if all conditions met
        if (std_error and std_error > 0 and 
            n_confounders <= 5 and 
            len([r for r in refutation_results if r.get('passed', False)]) >= 2):
            confidence = "high"
        
        return confidence, assumptions, warnings
    
    def _interpret_association(self, corr: float, pval: float) -> str:
        """Interpret correlation magnitude and significance."""
        # Magnitude
        if abs(corr) < 0.3:
            strength = "weak"
        elif abs(corr) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if corr > 0 else "negative"
        
        # Significance
        significance = "statistically significant" if pval < 0.05 else "not statistically significant"
        
        return f"{strength.capitalize()} {direction} association ({significance}, p={pval:.4f})"
