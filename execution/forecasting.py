"""
Time series forecasting module.

Implements ARIMA, ARIMAX, SARIMAX with exogenous variables, model selection, and scenario simulation.
Enhanced with frequency detection, stationarity testing, and standardized outputs.
"""
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf
from utils.helpers import get_logger

logger = get_logger(__name__)


class ForecastingEngine:
    """Time series forecasting with ARIMAX support and model selection."""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
    
    def forecast_metric(
        self,
        metric_col: str,
        time_col: str,
        exog_cols: Optional[List[str]] = None,  # Improvement #1: ARIMAX support
        periods: int = 12,
        model_type: str = "auto"  # auto, arima, sarimax, ets
    ) -> Dict[str, Any]:
        """
        Forecast a metric into the future with optional exogenous variables.
        
        Improvements:
        - ARIMAX/SARIMAX with exogenous variables
        - Frequency inference from time column
        - Stationarity testing
        - Model comparison and selection
        
        Args:
            metric_col: Column to forecast
            time_col: Time column
            exog_cols: Optional list of exogenous variable columns (for ARIMAX/SARIMAX)
            periods: Number of periods to forecast
            model_type: 'auto', 'arima', 'sarimax', 'ets'
            
        Returns:
            Dictionary with forecast results and metadata
        """
        logger.info(f"Forecasting {metric_col} for {periods} periods using {model_type}")
        if exog_cols:
            logger.info(f"Using exogenous variables: {exog_cols}")
        
        try:
            # Prepare data
            df_sorted = self.df.sort(time_col)
            values = df_sorted[metric_col].drop_nulls().to_numpy()
            
            if len(values) < 10:
                return {
                    "error": "Insufficient data for forecasting (need at least 10 points)",
                    "forecast_values": [],
                    "model_confidence": 0.0
                }
            
            # Improvement #2: Infer frequency from time column
            freq_info = self._infer_frequency(df_sorted, time_col)
            logger.info(f"Detected frequency: {freq_info['frequency']} (confidence: {freq_info['confidence']:.2f})")
            
            # Improvement #1: Prepare exogenous variables if provided
            exog_train = None
            exog_forecast = None
            if exog_cols:
                exog_train = df_sorted.select(exog_cols).drop_nulls().to_numpy()
                # For forecast, use last known values (can be enhanced with projection)
                exog_forecast = np.tile(exog_train[-1, :], (periods, 1))
                logger.info(f"Exogenous variables prepared: {exog_train.shape}")
            
            # Improvement #3: Stationarity test
            stationarity_result = self._test_stationarity(values)
            logger.info(f"Stationarity: {stationarity_result['is_stationary']} (ADF p-value: {stationarity_result['adf_pvalue']:.4f})")
            
            # Determine differencing order based on stationarity
            d_order = 0 if stationarity_result['is_stationary'] else 1
            
            # Improvement #4: Model comparison and selection
            if model_type == "auto":
                best_model, best_result = self._select_best_model(
                    values, exog_train, d_order, freq_info, periods, exog_forecast
                )
                logger.info(f"Auto-selected model: {best_model} (AIC: {best_result['aic']:.2f})")
            else:
                best_result = self._fit_single_model(
                    model_type, values, exog_train, d_order, freq_info, periods, exog_forecast
                )
                best_model = model_type
            
            # Improvement #6: Standardize output schema
            if "error" not in best_result:
                # Calculate trend direction
                forecast_values = best_result["forecast_values"]
                trend_direction = self._determine_trend(forecast_values)
                
                # Assess model confidence
                model_confidence = self._assess_confidence(
                    best_result, stationarity_result, len(values)
                )
                
                # Add metadata
                best_result.update({
                    "trend_direction": trend_direction,
                    "model_confidence": model_confidence,
                    "model_selected": best_model,
                    "frequency": freq_info['frequency'],
                    "is_stationary": stationarity_result['is_stationary'],
                    "exog_variables": exog_cols if exog_cols else [],
                    "notes": self._generate_notes(best_result, stationarity_result, freq_info)
                })
            
            logger.info(f"Forecast complete: {best_result.get('model_selected', 'unknown')}")
            return best_result
            
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            return {
                "error": str(e),
                "forecast_values": [],
                "model_confidence": 0.0
            }
    
    def simulate_scenario(
        self,
        metric_col: str,
        time_col: str,
        exog_col: str,
        delta_percent: float,
        periods: int = 12
    ) -> Dict[str, Any]:
        """
        Improvement #5: Scenario simulation via ARIMAX.
        
        Simulates impact of changing an exogenous variable.
        Example: "What if ad_spend increases by 20%?"
        
        Args:
            metric_col: Target metric to forecast
            time_col: Time column
            exog_col: Exogenous variable to modify
            delta_percent: Percentage change in exogenous variable (e.g., 20 for +20%)
            periods: Forecast horizon
            
        Returns:
            Dictionary with baseline, scenario forecast, and impact
        """
        logger.info(f"Simulating scenario: {exog_col} {delta_percent:+.1f}% impact on {metric_col}")
        
        try:
            # Baseline forecast
            baseline = self.forecast_metric(
                metric_col, time_col, exog_cols=[exog_col], periods=periods, model_type="auto"
            )
            
            if "error" in baseline:
                return baseline
            
            # Get current exog value
            df_sorted = self.df.sort(time_col)
            current_exog = df_sorted[exog_col].drop_nulls()[-1]
            delta_value = current_exog * (delta_percent / 100.0)
            new_exog = current_exog + delta_value
            
            # Prepare modified exogenous variables for scenario
            values = df_sorted[metric_col].drop_nulls().to_numpy()
            exog_train = df_sorted.select([exog_col]).drop_nulls().to_numpy()
            exog_scenario = np.full((periods, 1), new_exog)
            
            # Fit model with exog
            freq_info = self._infer_frequency(df_sorted, time_col)
            stationarity = self._test_stationarity(values)
            d_order = 0 if stationarity['is_stationary'] else 1
            
            # Use SARIMAX for scenario
            model = SARIMAX(
                values,
                exog=exog_train,
                order=(1, d_order, 1),
                seasonal_order=(1, d_order, 1, freq_info.get('seasonal_period', 12)),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted = model.fit(disp=False)
            
            # Forecast with modified exog
            scenario_forecast = fitted.forecast(steps=periods, exog=exog_scenario)
            
            # Calculate impact
            baseline_total = sum(baseline["forecast_values"])
            scenario_total = sum(scenario_forecast)
            impact = scenario_total - baseline_total
            impact_percent = (impact / baseline_total) * 100 if baseline_total != 0 else 0
            
            result = {
                "baseline_forecast": baseline["forecast_values"],
                "scenario_forecast": scenario_forecast.tolist(),
                "exog_variable": exog_col,
                "current_value": float(current_exog),
                "scenario_value": float(new_exog),
                "delta_percent": delta_percent,
                "impact_absolute": float(impact),
                "impact_percent": float(impact_percent),
                "interpretation": f"If {exog_col} changes by {delta_percent:+.1f}%, {metric_col} changes by {impact_percent:+.1f}%",
                "model_confidence": baseline.get("model_confidence", 0.5)
            }
            
            logger.info(f"Scenario impact: {impact_percent:+.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Scenario simulation failed: {e}")
            return {
                "error": str(e),
                "impact_absolute": 0.0,
                "impact_percent": 0.0
            }
    
    def _infer_frequency(self, df_sorted: pl.DataFrame, time_col: str) -> Dict[str, Any]:
        """
        Improvement #2: Infer frequency from time column.
        
        Detects daily, weekly, monthly patterns and gaps.
        """
        try:
            # Convert to pandas for datetime inference
            time_values = df_sorted[time_col].to_pandas()
            
            if pd.api.types.is_datetime64_any_dtype(time_values):
                # Calculate median time diff
                time_diff = time_values.diff().median()
                
                # Infer frequency
                if time_diff <= pd.Timedelta(days=1):
                    frequency = "daily"
                    seasonal_period = 7
                elif time_diff <= pd.Timedelta(days=7):
                    frequency = "weekly"
                    seasonal_period = 52
                elif time_diff <= pd.Timedelta(days=31):
                    frequency = "monthly"
                    seasonal_period = 12
                else:
                    frequency = "unknown"
                    seasonal_period = 12
                
                # Detect gaps
                gaps = (time_values.diff() > 2 * time_diff).sum()
                
                return {
                    "frequency": frequency,
                    "seasonal_period": seasonal_period,
                    "median_gap": str(time_diff),
                    "gaps_detected": int(gaps),
                    "confidence": 0.9 if gaps < 3 else 0.6
                }
            else:
                # Non-datetime, assume sequential
                return {
                    "frequency": "sequential",
                    "seasonal_period": 12,
                    "confidence": 0.5
                }
        
        except Exception as e:
            logger.warning(f"Frequency inference failed: {e}")
            return {
                "frequency": "unknown",
                "seasonal_period": 12,
                "confidence": 0.3
            }
    
    def _test_stationarity(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Improvement #3: Stationarity testing using Augmented Dickey-Fuller.
        
        Determines if differencing is needed.
        """
        try:
            adf_result = adfuller(values, autolag='AIC')
            
            return {
                "is_stationary": adf_result[1] < 0.05,  # p-value < 0.05
                "adf_statistic": float(adf_result[0]),
                "adf_pvalue": float(adf_result[1]),
                "critical_values": {k: float(v) for k, v in adf_result[4].items()}
            }
        
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}, assuming non-stationary")
            return {
                "is_stationary": False,
                "adf_pvalue": 1.0
            }
    
    def _select_best_model(
        self, values, exog_train, d_order, freq_info, periods, exog_forecast
    ) -> tuple[str, Dict[str, Any]]:
        """
        Improvement #4: Model comparison and selection.
        
        Tries ARIMA, SARIMAX (if exog), ETS and picks best by AIC.
        """
        models_to_try = []
        
        # ARIMA
        models_to_try.append(("arima", self._fit_single_model(
            "arima", values, None, d_order, freq_info, periods, None
        )))
        
        # SARIMAX (if exog provided)
        if exog_train is not None:
            models_to_try.append(("sarimax", self._fit_single_model(
                "sarimax", values, exog_train, d_order, freq_info, periods, exog_forecast
            )))
        
        # ETS (exponential smoothing) - no exog support
        try:
            ets_result = self._fit_ets(values, periods)
            if "error" not in ets_result:
                models_to_try.append(("ets", ets_result))
        except Exception as e:
            logger.warning(f"ETS failed: {e}")
        
        # Select best by AIC
        valid_models = [(name, result) for name, result in models_to_try if "error" not in result]
        
        if not valid_models:
            return "arima", {"error": "All models failed", "forecast_values": []}
        
        best_name, best_result = min(valid_models, key=lambda x: x[1].get("aic", float('inf')))
        
        return best_name, best_result
    
    def _fit_single_model(
        self, model_type, values, exog_train, d_order, freq_info, periods, exog_forecast
    ) -> Dict[str, Any]:
        """Fit a single model type."""
        try:
            if model_type == "sarimax":
                seasonal_period = freq_info.get('seasonal_period', 12)
                model = SARIMAX(
                    values,
                    exog=exog_train,
                    order=(1, d_order, 1),
                    seasonal_order=(1, d_order, 1, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:  # arima
                model = ARIMA(values, order=(1, d_order, 1))
            
            fitted = model.fit(disp=False)
            
            # Generate forecast
            if model_type == "sarimax" and exog_forecast is not None:
                forecast = fitted.forecast(steps=periods, exog=exog_forecast)
            else:
                forecast = fitted.forecast(steps=periods)
            
            # Confidence intervals
            forecast_result = fitted.get_forecast(steps=periods, exog=exog_forecast if model_type == "sarimax" else None)
            conf_int = forecast_result.conf_int()
            
            return {
                "forecast_values": forecast.tolist(),
                "confidence_lower": conf_int.iloc[:, 0].tolist() if hasattr(conf_int, 'iloc') else [],
                "confidence_upper": conf_int.iloc[:, 1].tolist() if hasattr(conf_int, 'iloc') else [],
                "model_type": model_type,
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
                "periods": periods
            }
        
        except Exception as e:
            logger.warning(f"{model_type} fit failed: {e}")
            return {"error": str(e)}
    
    def _fit_ets(self, values, periods) -> Dict[str, Any]:
        """Fit Exponential Smoothing (ETS) model."""
        try:
            model = ExponentialSmoothing(
                values,
                seasonal_periods=12,
                trend='add',
                seasonal='add'
            )
            fitted = model.fit()
            forecast = fitted.forecast(steps=periods)
            
            return {
                "forecast_values": forecast.tolist(),
                "confidence_lower": [],  # ETS doesn't provide conf intervals easily
                "confidence_upper": [],
                "model_type": "ets",
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
                "periods": periods
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def _determine_trend(self, forecast_values: List[float]) -> str:
        """Improvement #6: Determine trend direction."""
        if len(forecast_values) < 2:
            return "flat"
        
        # Calculate slope
        x = np.arange(len(forecast_values))
        y = np.array(forecast_values)
        slope = np.polyfit(x, y, 1)[0]
        
        # Relative to mean
        mean_val = np.mean(y)
        slope_percent = (slope / mean_val) * 100 if mean_val != 0 else 0
        
        if slope_percent > 5:
            return "up"
        elif slope_percent < -5:
            return "down"
        else:
            return "flat"
    
    def _assess_confidence(self, result, stationarity, n_samples) -> float:
        """Improvement #6: Assess model confidence."""
        confidence = 0.5  # Base
        
        # Boost if stationary
        if stationarity.get('is_stationary', False):
            confidence += 0.2
        
        # Boost for sufficient data
        if n_samples > 50:
            confidence += 0.2
        elif n_samples > 30:
            confidence += 0.1
        
        # Penalize high AIC relative to sample size
        aic = result.get('aic', 0)
        if aic > n_samples * 2:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_notes(self, result, stationarity, freq_info) -> str:
        """Improvement #6: Generate interpretive notes."""
        notes = []
        
        if not stationarity.get('is_stationary', False):
            notes.append("Series was non-stationary, differencing applied")
        
        if freq_info.get('gaps_detected', 0) > 2:
            notes.append(f"{freq_info['gaps_detected']} time gaps detected, may affect accuracy")
        
        if result.get('aic', 0) > 1000:
            notes.append("High model complexity, interpret with caution")
        
        return "; ".join(notes) if notes else "Model fit is reasonable"
    
    def detect_seasonality(self, metric_col: str, time_col: str) -> Dict[str, Any]:
        """Improved seasonality detection with confidence bands."""
        try:
            df_sorted = self.df.sort(time_col)
            values = df_sorted[metric_col].drop_nulls().to_numpy()
            
            if len(values) < 24:
                return {"seasonal": False, "reason": "Insufficient data"}
            
            # Autocorrelation with proper significance testing
            autocorr = acf(values, nlags=min(20, len(values) // 2), alpha=0.05)
            acf_values = autocorr[0] if isinstance(autocorr, tuple) else autocorr
            
            # Confidence band (approximate)
            conf_band = 1.96 / np.sqrt(len(values))
            
            # Find significant lags beyond confidence band
            significant_lags = np.where(np.abs(acf_values[1:]) > conf_band)[0]
            
            return {
                "seasonal": len(significant_lags) > 0,
                "potential_periods": significant_lags.tolist() if len(significant_lags) > 0 else [],
                "autocorrelation": acf_values[:12].tolist(),
                "confidence_band": float(conf_band)
            }
            
        except Exception as e:
            logger.error(f"Seasonality detection failed: {e}")
            return {"seasonal": False, "error": str(e)}
