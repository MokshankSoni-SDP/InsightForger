"""
Robust Data Loader Module.

Handles data ingestion with automated cleaning:
- Strips whitespace
- Removes currency symbols and thousands separators
- Safely converts numeric columns
- Normalizes column names
"""
import polars as pl
import re
from typing import Tuple, List, Dict
from utils.helpers import get_logger

logger = get_logger(__name__)

class RobustLoader:
    """Ingests and cleans tabular data robustness."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.preprocessing_log: List[str] = []
        
    def load_and_clean(self) -> pl.DataFrame:
        """
        Load CSV and apply robust cleaning strategies.
        
        Returns:
            Cleaned polars DataFrame
        """
        try:
            # 1. Load data
            # Use strict=False (ignore_errors=True) to handle mixed types by setting to null
            # READ ALL AS STRING ("infer_schema_length=0") to prevent aggressive Polars inference
            # getting in the way of our strict 70% threshold.
            df = pl.read_csv(
                self.file_path, 
                infer_schema_length=0, 
                ignore_errors=True, 
                truncate_ragged_lines=True
            ) 
            logger.info(f"Loaded raw data (strings): {df.shape}")
            
            # 2. Normalize headers
            df = self._normalize_headers(df)
            
            # 3. Clean and Cast Columns (numeric AND date)
            cleaned_df = self._clean_and_cast_columns(df)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise e

    def _normalize_headers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Snake_case column names and remove special chars."""
        new_cols = []
        for col in df.columns:
            clean = col.lower().strip()
            clean = re.sub(r'[^a-z0-9]', '_', clean)
            clean = re.sub(r'_{2,}', '_', clean)
            clean = clean.strip('_')
            if not clean: clean = f"col_{len(new_cols)}"
            new_cols.append(clean)
        return df.rename(dict(zip(df.columns, new_cols)))

    def _clean_and_cast_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean string columns that look like numbers using strict validation.
        Also attempts Date/Datetime inference.
        
        Logic:
        1. For each string column:
           a. Check Date inference (e.g. Rate > 70%).
           b. If not Date, clean (remove currency, commas) and Check Numeric inference (Rate > 70%).
        2. Replace if successful.
        """
        exprs = []
        
        for col in df.columns:
            # We know everything is Utf8 initially because infer_schema_length=0
            # But just in case
            if df[col].dtype != pl.Utf8:
                exprs.append(pl.col(col))
                continue
                
            series = df[col]
            if series.null_count() == len(series):
                exprs.append(pl.col(col))
                continue

            original_non_null = series.drop_nulls().len()
            if original_non_null == 0:
                 exprs.append(pl.col(col))
                 continue

            # --- Attempt 1: Date/Datetime with Multi-Format Fallback ---
            # Try multiple common date formats to handle DD-MM-YYYY, MM/DD/YYYY, etc.
            date_formats = [
                None,           # Polars default (YYYY-MM-DD, ISO 8601)
                "%d-%m-%Y",     # DD-MM-YYYY (common in Europe, India)
                "%d/%m/%Y",     # DD/MM/YYYY
                "%m/%d/%Y",     # MM/DD/YYYY (US format)
                "%Y/%m/%d",     # YYYY/MM/DD
                "%d-%b-%Y",     # DD-Mon-YYYY (e.g., 01-Jan-2024)
                "%d %b %Y",     # DD Mon YYYY (e.g., 01 Jan 2024)
            ]
            
            best_date_rate = 0.0
            best_format = None
            date_parsed = False
            
            for fmt in date_formats:
                try:
                    if fmt is None:
                        # Try Polars default inference
                        date_cast = series.str.to_datetime(strict=False)
                    else:
                        # Try specific format
                        date_cast = series.str.to_datetime(format=fmt, strict=False)
                    
                    valid_dates = date_cast.drop_nulls().len()
                    date_rate = valid_dates / original_non_null
                    
                    # Keep track of best format
                    if date_rate > best_date_rate:
                        best_date_rate = date_rate
                        best_format = fmt
                    
                    # If we found a good match (>= 70%), use it immediately
                    if date_rate >= 0.7:
                        format_name = fmt if fmt else "ISO 8601 (YYYY-MM-DD)"
                        self.preprocessing_log.append(
                            f"Successfully cast '{col}' to Datetime using format '{format_name}' (Rate: {date_rate:.1%})"
                        )
                        if fmt is None:
                            exprs.append(pl.col(col).str.to_datetime(strict=False).alias(col))
                        else:
                            exprs.append(pl.col(col).str.to_datetime(format=fmt, strict=False).alias(col))
                        date_parsed = True
                        break  # Found a good format, move to next column
                        
                except Exception:
                    # This format didn't work, try next one
                    continue
            
            # If date was successfully parsed, skip to next column
            if date_parsed:
                continue
            
            # No format worked well enough (< 70%)
            # If we found any partial match, log it
            if best_date_rate > 0:
                logger.debug(
                    f"Column '{col}' has some date-like values (best rate: {best_date_rate:.1%}) "
                    f"but below 70% threshold. Keeping as String."
                )
            
            # --- Attempt 2: Numeric ---
            # Clean: strip whitespace, remove currency symbols, commas, percent
            clean_expr = (
                pl.col(col)
                .str.strip_chars()
                .str.replace_all(r'[$£€₹,%\s]', '') # Remove symbols and whitespace
            )
            
            # Try Cast
            cast_expr = clean_expr.cast(pl.Float64, strict=False)
            
            # Evaluate strictly
            cleaned_s = (
                series.str.strip_chars()
                .str.replace_all(r'[$£€₹,%\s]', '')
                .cast(pl.Float64, strict=False)
            )
            
            valid_numeric = cleaned_s.drop_nulls().len()
            numeric_rate = valid_numeric / original_non_null
            
            if numeric_rate >= 0.7:
                self.preprocessing_log.append(f"Successfully cast '{col}' to Float64 (Rate: {numeric_rate:.1%})")
                exprs.append(cast_expr.alias(col))
            else:
                if valid_numeric > 0:
                    logger.warning(
                        f"Column '{col}' looks numeric but failed strict threshold "
                        f"(Rate: {numeric_rate:.1%} < 70%). Keeping as String."
                    )
                exprs.append(pl.col(col))
                
        return df.select(exprs)
