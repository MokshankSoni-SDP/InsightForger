"""
Anomaly detection module.

Implements multiple anomaly detection methods with context preservation,
severity scoring, and cross-method consensus.
Enhanced to provide actionable anomaly insights.
"""
import polars as pl
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import IsolationForest
from utils.helpers import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Anomaly detection with context preservation and consensus validation."""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
    
    def detect_anomalies_iqr(
        self,
        column: str,
        time_col: Optional[str] = None,
        context_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using IQR method.
        
        Improvements:
        - Preserves row indices and time context
        - Returns severity scores
        - Includes anomaly classification hints
        
        Args:
            column: Column to analyze
            time_col: Optional time column for temporal context
            context_cols: Optional categorical columns for grouping context
            
        Returns:
            Dictionary with anomaly details including indices and severity
        """
        logger.info(f"Detecting IQR anomalies in {column}")
        
        try:
            # Improvement #1: Preserve context - work with full dataframe
            df_clean = self.df.filter(pl.col(column).is_not_null())
            values = df_clean[column].to_numpy()
            
            if len(values) < 10:
                return {
                    "error": "Insufficient data for anomaly detection",
                    "anomalies_found": 0
                }
            
            # Calculate IQR
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Identify anomalies
            anomaly_mask = (values < lower_bound) | (values > upper_bound)
            anomaly_indices = np.where(anomaly_mask)[0]
            
            # Improvement #3: Calculate severity scores
            severity_scores = []
            for idx in anomaly_indices:
                val = values[idx]
                if val > upper_bound:
                    severity = (val - upper_bound) / iqr if iqr > 0 else 0
                else:
                    severity = (lower_bound - val) / iqr if iqr > 0 else 0
                severity_scores.append(float(severity))
            
            # Improvement #1: Extract context for anomalies
            anomaly_details = self._extract_anomaly_context(
                df_clean, anomaly_indices, column, values, time_col, context_cols, severity_scores
            )
            
            # Improvement #5: Classify anomalies
            anomaly_details = self._classify_anomalies(anomaly_details, values, upper_bound, lower_bound)
            
            result = {
                "method": "iqr",
                "column": column,
                "anomalies_found": len(anomaly_indices),
                "anomaly_rate": len(anomaly_indices) / len(values),
                "bounds": {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                    "iqr": float(iqr)
                },
                "anomaly_details": anomaly_details,
                "summary_stats": {
                    "mean_severity": float(np.mean(severity_scores)) if severity_scores else 0.0,
                    "max_severity": float(np.max(severity_scores)) if severity_scores else 0.0
                }
            }
            
            logger.info(f"Found {len(anomaly_indices)} anomalies ({result['anomaly_rate']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"IQR anomaly detection failed: {e}")
            return {"error": str(e), "anomalies_found": 0}
    
    def detect_anomalies_zscore(
        self,
        column: str,
        threshold: float = 3.0,
        time_col: Optional[str] = None,
        context_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using Z-score method.
        
        Note: Assumes roughly normal distribution.
        
        Improvements:
        - Preserves context
        - Severity = absolute z-score
        - Classification hints
        """
        logger.info(f"Detecting Z-score anomalies in {column} (threshold={threshold})")
        
        try:
            df_clean = self.df.filter(pl.col(column).is_not_null())
            values = df_clean[column].to_numpy()
            
            if len(values) < 10:
                return {"error": "Insufficient data", "anomalies_found": 0}
            
            # Calculate Z-scores
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return {"error": "Zero standard deviation", "anomalies_found": 0}
            
            z_scores = np.abs((values - mean) / std)
            anomaly_mask = z_scores > threshold
            anomaly_indices = np.where(anomaly_mask)[0]
            
            # Improvement #3: Severity = Z-score magnitude
            severity_scores = [float(z_scores[idx]) for idx in anomaly_indices]
            
            # Improvement #1: Extract context
            anomaly_details = self._extract_anomaly_context(
                df_clean, anomaly_indices, column, values, time_col, context_cols, severity_scores
            )
            
            # Improvement #5: Classify
            upper_bound = mean + threshold * std
            lower_bound = mean - threshold * std
            anomaly_details = self._classify_anomalies(anomaly_details, values, upper_bound, lower_bound)
            
            result = {
                "method": "zscore",
                "column": column,
                "anomalies_found": len(anomaly_indices),
                "anomaly_rate": len(anomaly_indices) / len(values),
                "threshold": threshold,
                "distribution": {
                    "mean": float(mean),
                    "std": float(std)
                },
                "anomaly_details": anomaly_details,
                "summary_stats": {
                    "mean_severity": float(np.mean(severity_scores)) if severity_scores else 0.0,
                    "max_severity": float(np.max(severity_scores)) if severity_scores else 0.0
                },
                "warning": "Z-score assumes normal distribution - may be inaccurate for skewed data"
            }
            
            logger.info(f"Found {len(anomaly_indices)} anomalies ({result['anomaly_rate']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Z-score anomaly detection failed: {e}")
            return {"error": str(e), "anomalies_found": 0}
    
    def detect_anomalies_isolation_forest(
        self,
        column: str,
        time_col: Optional[str] = None,
        context_cols: Optional[List[str]] = None,
        contamination: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest.
        
        Improvements:
        - Adaptive contamination based on IQR anomaly rate
        - Context preservation
        - Severity from anomaly scores
        """
        logger.info(f"Detecting Isolation Forest anomalies in {column}")
        
        try:
            df_clean = self.df.filter(pl.col(column).is_not_null())
            values = df_clean[column].to_numpy().reshape(-1, 1)
            
            if len(values) < 10:
                return {"error": "Insufficient data", "anomalies_found": 0}
            
            # Improvement #2: Adaptive contamination
            if contamination is None:
                contamination = self._estimate_contamination(df_clean[column].to_numpy())
                logger.info(f"Adaptive contamination: {contamination:.3f}")
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            predictions = iso_forest.fit_predict(values)
            
            # Get anomaly scores (more negative = more anomalous)
            anomaly_scores = iso_forest.score_samples(values)
            
            # Identify anomalies (-1 = anomaly, 1 = normal)
            anomaly_mask = predictions == -1
            anomaly_indices = np.where(anomaly_mask)[0]
            
            # Improvement #3: Severity from negative anomaly scores
            severity_scores = [
                float(-anomaly_scores[idx])  # Negate so higher = more severe
                for idx in anomaly_indices
            ]
            
            # Improvement #1: Extract context
            values_flat = values.flatten()
            anomaly_details = self._extract_anomaly_context(
                df_clean, anomaly_indices, column, values_flat, time_col, context_cols, severity_scores
            )
            
            # Improvement #5: Classify (use simple thresholds)
            mean_val = np.mean(values_flat)
            std_val = np.std(values_flat)
            upper_bound = mean_val + 2 * std_val
            lower_bound = mean_val - 2 * std_val
            anomaly_details = self._classify_anomalies(anomaly_details, values_flat, upper_bound, lower_bound)
            
            result = {
                "method": "isolation_forest",
                "column": column,
                "anomalies_found": len(anomaly_indices),
                "anomaly_rate": len(anomaly_indices) / len(values),
                "contamination": contamination,
                "anomaly_details": anomaly_details,
                "summary_stats": {
                    "mean_severity": float(np.mean(severity_scores)) if severity_scores else 0.0,
                    "max_severity": float(np.max(severity_scores)) if severity_scores else 0.0
                }
            }
            
            logger.info(f"Found {len(anomaly_indices)} anomalies ({result['anomaly_rate']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            return {"error": str(e), "anomalies_found": 0}
    
    def detect_consensus_anomalies(
        self,
        column: str,
        time_col: Optional[str] = None,
        context_cols: Optional[List[str]] = None,
        min_methods: int = 2
    ) -> Dict[str, Any]:
        """
        Improvement #4: Cross-detector consensus.
        
        Runs multiple detection methods and flags anomalies where
        ≥ min_methods agree.
        
        Args:
            column: Column to analyze
            time_col: Optional time column
            context_cols: Optional context columns
            min_methods: Minimum methods that must agree (default: 2)
            
        Returns:
            Dictionary with consensus anomalies and confidence levels
        """
        logger.info(f"Running consensus anomaly detection on {column}")
        
        try:
            # Run all three methods
            iqr_result = self.detect_anomalies_iqr(column, time_col, context_cols)
            zscore_result = self.detect_anomalies_zscore(column, time_col=time_col, context_cols=context_cols)
            iforest_result = self.detect_anomalies_isolation_forest(column, time_col, context_cols)
            
            # Collect anomaly indices from each method
            methods_results = []
            
            if "error" not in iqr_result and iqr_result["anomalies_found"] > 0:
                iqr_indices = set(a["index"] for a in iqr_result["anomaly_details"])
                methods_results.append(("iqr", iqr_indices, iqr_result))
            
            if "error" not in zscore_result and zscore_result["anomalies_found"] > 0:
                zscore_indices = set(a["index"] for a in zscore_result["anomaly_details"])
                methods_results.append(("zscore", zscore_indices, zscore_result))
            
            if "error" not in iforest_result and iforest_result["anomalies_found"] > 0:
                iforest_indices = set(a["index"] for a in iforest_result["anomaly_details"])
                methods_results.append(("isolation_forest", iforest_indices, iforest_result))
            
            if len(methods_results) < min_methods:
                return {
                    "error": f"Fewer than {min_methods} methods succeeded",
                    "methods_run": len(methods_results)
                }
            
            # Find consensus
            all_indices = set()
            for _, indices, _ in methods_results:
                all_indices.update(indices)
            
            consensus_anomalies = []
            for idx in all_indices:
                # Count how many methods flagged this index
                methods_flagged = [name for name, indices, _ in methods_results if idx in indices]
                confidence = "high" if len(methods_flagged) >= 3 else "medium" if len(methods_flagged) >= 2 else "low"
                
                if len(methods_flagged) >= min_methods:
                    # Get details from first method that found it
                    for name, indices, result in methods_results:
                        if idx in indices:
                            detail = next((a for a in result["anomaly_details"] if a["index"] == idx), None)
                            if detail:
                                detail["consensus_confidence"] = confidence
                                detail["methods_agreed"] = methods_flagged
                                detail["agreement_count"] = len(methods_flagged)
                                consensus_anomalies.append(detail)
                                break
            
            # Sort by severity
            consensus_anomalies.sort(key=lambda x: x.get("severity", 0), reverse=True)
            
            result = {
                "method": "consensus",
                "column": column,
                "min_methods_required": min_methods,
                "methods_run": [name for name, _, _ in methods_results],
                "consensus_anomalies_found": len(consensus_anomalies),
                "anomaly_details": consensus_anomalies,
                "individual_results": {
                    name: {
                        "anomalies_found": len(indices),
                        "anomaly_rate": result["anomaly_rate"]
                    }
                    for name, indices, result in methods_results
                }
            }
            
            logger.info(f"Consensus: {len(consensus_anomalies)} anomalies with ≥{min_methods} method agreement")
            return result
            
        except Exception as e:
            logger.error(f"Consensus detection failed: {e}")
            return {"error": str(e)}
    
    def _estimate_contamination(self, values: np.ndarray) -> float:
        """
        Improvement #2: Estimate contamination adaptively using IQR.
        
        Returns a conservative estimate based on IQR outlier rate.
        """
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = np.sum((values < lower_bound) | (values > upper_bound))
        iqr_rate = outliers / len(values)
        
        # Bounded between 0.01 and 0.15
        contamination = max(0.01, min(0.15, iqr_rate))
        
        return contamination
    
    def _extract_anomaly_context(
        self,
        df_clean: pl.DataFrame,
        anomaly_indices: np.ndarray,
        column: str,
        values: np.ndarray,
        time_col: Optional[str],
        context_cols: Optional[List[str]],
        severity_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Improvement #1: Extract full context for each anomaly.
        
        Returns list of anomaly details with indices, values, time, and context.
        """
        anomaly_details = []
        
        for i, idx in enumerate(anomaly_indices):
            detail = {
                "index": int(idx),
                "value": float(values[idx]),
                "severity": severity_scores[i] if i < len(severity_scores) else 0.0
            }
            
            # Add time context if available
            if time_col and time_col in df_clean.columns:
                time_val = df_clean[time_col][idx]
                detail["time"] = str(time_val)
            
            # Add categorical context
            if context_cols:
                for ctx_col in context_cols:
                    if ctx_col in df_clean.columns:
                        detail[ctx_col] = str(df_clean[ctx_col][idx])
            
            anomaly_details.append(detail)
        
        return anomaly_details
    
    def _classify_anomalies(
        self,
        anomaly_details: List[Dict[str, Any]],
        all_values: np.ndarray,
        upper_bound: float,
        lower_bound: float
    ) -> List[Dict[str, Any]]:
        """
        Improvement #5: Add simple anomaly classification hints.
        
        Classifies as: spike, drop, or data_issue
        """
        for detail in anomaly_details:
            val = detail["value"]
            
            if val > upper_bound:
                # Spike
                if val > upper_bound * 2:
                    detail["classification"] = "extreme_spike"
                    detail["hint"] = "Possible event or data error"
                else:
                    detail["classification"] = "spike"
                    detail["hint"] = "Above normal range"
            
            elif val < lower_bound:
                # Drop
                if val <= 0 or val < lower_bound / 2:
                    detail["classification"] = "extreme_drop"
                    detail["hint"] = "Possible failure or dead stock"
                else:
                    detail["classification"] = "drop"
                    detail["hint"] = "Below normal range"
            
            # Check for zero (data issue)
            if val == 0:
                detail["classification"] = "zero_value"
                detail["hint"] = "Data issue or actual zero"
        
        return anomaly_details
