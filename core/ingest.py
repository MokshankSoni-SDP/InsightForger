"""
Data ingestion and cleaning module.

Handles CSV upload, validation, and automated cleaning.
"""
import polars as pl
from typing import Optional, Tuple
from pathlib import Path
from utils.helpers import normalize_column_name, get_logger

logger = get_logger(__name__)


class DataIngestor:
    """Handles CSV loading and automatic cleaning."""
    
    def __init__(self):
        self.df: Optional[pl.DataFrame] = None
        self.original_df: Optional[pl.DataFrame] = None
        self.cleaning_log: list = []
    
    def load_csv(self, file_path: str) -> pl.DataFrame:
        """
        Load CSV file with automatic encoding detection.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading CSV from: {file_path}")
        
        try:
            # Try UTF-8 first
            self.df = pl.read_csv(file_path, ignore_errors=True)
            logger.info(f"Successfully loaded CSV with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise ValueError(f"Could not load CSV file: {e}")
        
        # Keep original copy
        self.original_df = self.df.clone()
        return self.df
    
    def clean_data(self) -> pl.DataFrame:
        """
        Auto-clean the loaded dataframe.
        
        Steps:
        1. Strip whitespace from string columns
        2. Normalize column names
        3. Parse dates
        4. Drop empty columns
        5. Basic validation
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        logger.info("Starting auto-clean process")
        
        # 1. Normalize column names
        original_cols = self.df.columns
        normalized_cols = [normalize_column_name(col) for col in original_cols]
        self.df.columns = normalized_cols
        self.cleaning_log.append(f"Normalized {len(original_cols)} column names")
        logger.info("Column names normalized")
        
        # 2. Strip whitespace from string columns
        for col in self.df.columns:
            if self.df[col].dtype == pl.Utf8:
                self.df = self.df.with_columns(
                    pl.col(col).str.strip_chars().alias(col)
                )
        self.cleaning_log.append("Stripped whitespace from text columns")
        logger.info("Whitespace stripped from text columns")
        
        # 3. Drop completely empty columns
        empty_cols = [col for col in self.df.columns if self.df[col].null_count() == len(self.df)]
        if empty_cols:
            self.df = self.df.drop(empty_cols)
            self.cleaning_log.append(f"Dropped {len(empty_cols)} empty columns: {empty_cols}")
            logger.info(f"Dropped {len(empty_cols)} empty columns")
        
        # 4. Auto-detect and parse date columns
        date_candidates = [col for col in self.df.columns if any(
            keyword in col.lower() for keyword in ["date", "time", "timestamp"]
        )]
        
        for col in date_candidates:
            if self.df[col].dtype == pl.Utf8:
                try:
                    self.df = self.df.with_columns(
                        pl.col(col).str.to_datetime(strict=False).alias(col)
                    )
                    self.cleaning_log.append(f"Parsed '{col}' as datetime")
                    logger.info(f"Parsed column '{col}' as datetime")
                except:
                    # If parsing fails, keep as string
                    pass
        
        # 5. Basic validation
        if len(self.df) == 0:
            raise ValueError("Dataset is empty after cleaning")
        
        if len(self.df.columns) == 0:
            raise ValueError("No columns remain after cleaning")
        
        self.cleaning_log.append(f"Final dataset: {len(self.df)} rows × {len(self.df.columns)} columns")
        logger.info(f"Cleaning complete. Final shape: {self.df.shape}")
        
        return self.df
    
    def get_sample(self, n: int = 5) -> pl.DataFrame:
        """Get a sample of the data."""
        if self.df is None:
            raise ValueError("No data loaded")
        return self.df.head(n)
    
    def get_cleaning_summary(self) -> str:
        """Get summary of cleaning operations."""
        return "\n".join(self.cleaning_log)
    
    def reset(self):
        """Reset to original data."""
        if self.original_df is not None:
            self.df = self.original_df.clone()
            self.cleaning_log = []
