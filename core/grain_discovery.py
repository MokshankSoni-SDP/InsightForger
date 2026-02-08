"""
Grain Discovery Module (Enhanced).

Determines the "unit of observation" in the dataset using dynamic recursive uniqueness search.
Fixes three critical flaws:
- Flaw A: Always detects temporal frequency (even for transactional data)
- Flaw B: No magic number thresholds (uses recursive uniqueness search)
- Flaw C: Semantic-aware dimension selection (uses Phase 0.3 profile)
"""
import polars as pl
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from utils.schemas import SemanticProfile
from utils.helpers import get_logger

logger = get_logger(__name__)


class DataGrain(Enum):
    """Types of data grain."""
    TRANSACTIONAL = "transactional"  # Multiple rows per time period
    TIMESERIES_DAILY = "timeseries_daily"  # One row per day
    TIMESERIES_WEEKLY = "timeseries_weekly"  # One row per week
    TIMESERIES_MONTHLY = "timeseries_monthly"  # One row per month
    SNAPSHOT = "snapshot"  # One row per entity (no time dimension)
    ROW_INDEX = "row_index"  # Event-level data (no unique key found)
    UNKNOWN = "unknown"


@dataclass
class GrainProfile:
    """Profile describing the grain of the dataset."""
    primary_grain: DataGrain
    temporal_grain: Optional[str]  # "daily", "weekly", "monthly", None
    temporal_frequency: Optional[str]  # Same as temporal_grain (for clarity)
    categorical_grain: Optional[List[str]]  # Dimension columns in grain
    composite_key: Optional[List[str]]  # The actual unique key columns
    is_unique: bool  # True if composite_key is 99%+ unique
    uniqueness_ratio: float  # Actual uniqueness ratio (0.0-1.0)
    row_represents: str  # Human-readable description
    aggregation_needed: bool  # True if data needs aggregation for time series
    avg_rows_per_period: Optional[float]  # Average rows per time period


class GrainDiscoverer:
    """Discovers the grain (unit of observation) of a dataset using semantic-aware recursive search."""
    
    def __init__(self, df: pl.DataFrame, profile: SemanticProfile):
        self.df = df
        self.profile = profile
    
    def discover_grain(self) -> GrainProfile:
        """
        Discover the grain of the dataset using dynamic recursive uniqueness search.
        
        Algorithm:
        1. Check if time column exists
        2. Test Date uniqueness (99%+)
        3. If not unique, find dimension columns (semantic-aware)
        4. Recursive uniqueness search: Date + Dim1, Date + Dim1 + Dim2, ...
        5. Always detect temporal frequency (even for transactional)
        
        Returns:
            GrainProfile describing the unit of observation
        """
        logger.info("Discovering data grain (dynamic recursive search)")
        
        # Step 1: Check if we have time columns
        time_cols = self.profile.time_columns
        
        if not time_cols:
            return self._discover_snapshot_grain()
        
        time_col = time_cols[0]
        
        # Step 2: ALWAYS detect temporal frequency (FIX FOR FLAW A)
        temporal_frequency = self._detect_temporal_frequency(time_col)
        logger.info(f"  Temporal frequency: {temporal_frequency}")
        
        # Step 3: Test Date uniqueness
        date_uniqueness = self._test_uniqueness([time_col])
        logger.info(f"  Date uniqueness: {date_uniqueness:.1%}")
        
        if date_uniqueness >= 0.99:
            # Pure time series - Date alone is unique
            return self._create_timeseries_grain(time_col, temporal_frequency, date_uniqueness)
        
        # Step 4: Find dimension columns (semantic-aware) (FIX FOR FLAW C)
        dimension_cols = self._find_dimension_columns()
        logger.info(f"  Found {len(dimension_cols)} dimension columns: {dimension_cols[:5]}")
        
        # Step 5: Recursive uniqueness search (FIX FOR FLAW B)
        composite_key, uniqueness = self._find_composite_key(time_col, dimension_cols)
        logger.info(f"  Composite key: {composite_key} (uniqueness: {uniqueness:.1%})")
        
        # Step 6: Determine grain type
        if uniqueness >= 0.99:
            # Transactional data with unique composite key
            return self._create_transactional_grain(
                time_col, temporal_frequency, composite_key, uniqueness
            )
        else:
            # Event-level data (no unique key found)
            return self._create_row_index_grain(
                time_col, temporal_frequency, composite_key, uniqueness
            )
    
    def _test_uniqueness(self, cols: List[str]) -> float:
        """
        Test uniqueness ratio for given columns.
        
        Handles edge case: Floating point IDs by force-casting to strings.
        
        Args:
            cols: List of column names to test
            
        Returns:
            Uniqueness ratio (0.0-1.0)
        """
        try:
            # Force-cast to strings to handle floating point IDs
            test_df = self.df.select([
                pl.col(c).cast(pl.Utf8).alias(c) for c in cols
            ])
            
            # Create composite key
            composite = test_df.select(
                pl.concat_str([pl.col(c) for c in cols], separator="||").alias("key")
            )
            
            unique_count = composite["key"].n_unique()
            total_count = len(composite)
            
            return unique_count / total_count if total_count > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error testing uniqueness for {cols}: {e}")
            return 0.0
    
    def _find_dimension_columns(self) -> List[str]:
        """
        Find dimension columns using semantic profile (Phase 0.3 output).
        
        Priority:
        1. Identifiers (semantic_guess = "identifier")
        2. Ordinal dimensions (is_ordinal = True)
        3. High-confidence dimensions (confidence >= 0.70)
        
        Excludes:
        - Low-cardinality categories (unique_ratio < 0.05)
        - Metrics (semantic_guess = "financial", "inventory", etc.)
        - Time columns (already handled separately)
        
        Returns:
            List of dimension column names (max 10 to avoid slow search)
        """
        dimensions = []
        
        # Priority 1: Identifiers
        for entity in self.profile.entities:
            if (entity.semantic_guess == "identifier" and 
                entity.confidence >= 0.70 and
                entity.column_name not in self.profile.time_columns):
                dimensions.append(entity.column_name)
                logger.debug(f"    Added identifier: {entity.column_name} (conf={entity.confidence:.2f})")
        
        # Priority 2: Ordinal dimensions
        for entity in self.profile.entities:
            if (entity.is_ordinal and 
                entity.column_name not in dimensions and
                entity.column_name not in self.profile.time_columns):
                dimensions.append(entity.column_name)
                logger.debug(f"    Added ordinal: {entity.column_name}")
        
        # Priority 3: High-confidence categorical dimensions
        for entity in self.profile.entities:
            if (entity.statistical_type == "categorical" and 
                entity.confidence >= 0.70 and
                entity.unique_ratio > 0.05 and  # Exclude low-cardinality
                entity.column_name not in dimensions and
                entity.column_name not in self.profile.time_columns):
                dimensions.append(entity.column_name)
                logger.debug(f"    Added categorical: {entity.column_name} (conf={entity.confidence:.2f}, ratio={entity.unique_ratio:.2f})")
        
        # Limit to top 10 to avoid slow search (edge case: 100 columns)
        return dimensions[:10]
    
    def _find_composite_key(self, time_col: str, dims: List[str]) -> Tuple[List[str], float]:
        """
        Recursively find composite key that achieves 99%+ uniqueness.
        
        Algorithm:
        1. Start with [time_col]
        2. Try [time_col, dim1]
        3. Try [time_col, dim1, dim2]
        4. Stop when uniqueness >= 0.99 or all dims exhausted
        
        Args:
            time_col: Time column name
            dims: List of dimension columns (ordered by priority)
            
        Returns:
            Tuple of (composite_key, uniqueness_ratio)
        """
        # Start with just time column (already tested, but for clarity)
        best_key = [time_col]
        best_uniqueness = self._test_uniqueness([time_col])
        
        if best_uniqueness >= 0.99:
            return best_key, best_uniqueness
        
        # Try adding dimensions one by one
        for i in range(len(dims)):
            test_key = [time_col] + dims[:i+1]
            uniqueness = self._test_uniqueness(test_key)
            
            logger.debug(f"    Testing {test_key}: {uniqueness:.1%}")
            
            if uniqueness > best_uniqueness:
                best_key = test_key
                best_uniqueness = uniqueness
            
            # Stop if we found a unique key
            if uniqueness >= 0.99:
                break
        
        return best_key, best_uniqueness
    
    def _create_timeseries_grain(
        self, time_col: str, temporal_frequency: str, uniqueness: float
    ) -> GrainProfile:
        """Create grain profile for pure time series data (Date alone is unique)."""
        # Determine primary grain type based on frequency
        if temporal_frequency == "daily":
            primary_grain = DataGrain.TIMESERIES_DAILY
        elif temporal_frequency == "weekly":
            primary_grain = DataGrain.TIMESERIES_WEEKLY
        elif temporal_frequency == "monthly":
            primary_grain = DataGrain.TIMESERIES_MONTHLY
        else:
            primary_grain = DataGrain.UNKNOWN
        
        return GrainProfile(
            primary_grain=primary_grain,
            temporal_grain=temporal_frequency,
            temporal_frequency=temporal_frequency,
            categorical_grain=None,
            composite_key=[time_col],
            is_unique=True,
            uniqueness_ratio=uniqueness,
            row_represents=f"One row per {temporal_frequency}",
            aggregation_needed=False,
            avg_rows_per_period=1.0
        )
    
    def _create_transactional_grain(
        self, time_col: str, temporal_frequency: str, 
        composite_key: List[str], uniqueness: float
    ) -> GrainProfile:
        """Create grain profile for transactional data (Date + Dims is unique)."""
        # Calculate avg rows per period
        try:
            date_counts = self.df.group_by(time_col).agg(pl.count().alias("row_count"))
            avg_rows = date_counts["row_count"].mean()
        except:
            avg_rows = None
        
        # Build row description
        key_str = " + ".join(composite_key)
        categorical_dims = [c for c in composite_key if c != time_col]
        
        return GrainProfile(
            primary_grain=DataGrain.TRANSACTIONAL,
            temporal_grain=temporal_frequency,  # ✅ FIX FOR FLAW A - NOW STORED
            temporal_frequency=temporal_frequency,
            categorical_grain=categorical_dims if categorical_dims else None,
            composite_key=composite_key,
            is_unique=True,
            uniqueness_ratio=uniqueness,
            row_represents=f"One transaction per {key_str} (avg {avg_rows:.0f} per {temporal_frequency})" if avg_rows else f"One transaction per {key_str}",
            aggregation_needed=True,
            avg_rows_per_period=avg_rows
        )
    
    def _create_row_index_grain(
        self, time_col: str, temporal_frequency: str,
        composite_key: List[str], uniqueness: float
    ) -> GrainProfile:
        """Create grain profile for event-level data (no unique key found)."""
        # Calculate avg rows per period
        try:
            date_counts = self.df.group_by(time_col).agg(pl.count().alias("row_count"))
            avg_rows = date_counts["row_count"].mean()
        except:
            avg_rows = None
        
        categorical_dims = [c for c in composite_key if c != time_col]
        
        return GrainProfile(
            primary_grain=DataGrain.ROW_INDEX,
            temporal_grain=temporal_frequency,  # ✅ Still store frequency
            temporal_frequency=temporal_frequency,
            categorical_grain=categorical_dims if categorical_dims else None,
            composite_key=composite_key,
            is_unique=False,  # Flag as non-unique
            uniqueness_ratio=uniqueness,
            row_represents=f"Event-level data (non-unique transactions, {uniqueness:.0%} unique)",
            aggregation_needed=True,
            avg_rows_per_period=avg_rows
        )
    
    def _discover_snapshot_grain(self) -> GrainProfile:
        """
        Discover grain for snapshot data (no time dimension).
        
        Uses semantic-aware dimension selection.
        """
        dimension_cols = self._find_dimension_columns()
        
        if dimension_cols:
            # Test uniqueness of dimension columns
            uniqueness = self._test_uniqueness(dimension_cols[:3])  # Test top 3
            
            if uniqueness >= 0.99:
                row_desc = f"One row per {', '.join(dimension_cols[:2])}"
                is_unique = True
            else:
                row_desc = f"Snapshot data ({uniqueness:.0%} unique by {', '.join(dimension_cols[:2])})"
                is_unique = False
        else:
            row_desc = "One row per entity (no dimensions found)"
            uniqueness = 0.0
            is_unique = False
        
        return GrainProfile(
            primary_grain=DataGrain.SNAPSHOT,
            temporal_grain=None,
            temporal_frequency=None,
            categorical_grain=dimension_cols[:3] if dimension_cols else None,
            composite_key=dimension_cols[:3] if dimension_cols else None,
            is_unique=is_unique,
            uniqueness_ratio=uniqueness,
            row_represents=row_desc,
            aggregation_needed=False,
            avg_rows_per_period=None
        )
    
    def _detect_temporal_frequency(self, time_col: str) -> str:
        """
        Detect the temporal frequency (daily, weekly, monthly).
        
        Uses median difference between consecutive dates for robustness.
        
        Args:
            time_col: Name of the time column
            
        Returns:
            Frequency string: "daily", "weekly", "monthly", "irregular", or "unknown"
        """
        try:
            dates = self.df[time_col].sort()
            
            # Calculate differences between consecutive dates
            diffs = dates.diff().drop_nulls()
            
            if len(diffs) == 0:
                return "unknown"
            
            # Get median difference (more robust than mean)
            median_diff = diffs.median()
            
            if median_diff is None:
                return "unknown"
            
            # Convert to days if it's a duration
            if hasattr(median_diff, 'days'):
                median_diff_days = median_diff.days
            else:
                # Assume it's already in days or a numeric type
                median_diff_days = float(median_diff)
            
            # Classify based on median difference
            if median_diff_days <= 1.5:
                return "daily"
            elif median_diff_days <= 8:
                return "weekly"
            elif median_diff_days <= 35:
                return "monthly"
            else:
                return "irregular"
        except Exception as e:
            logger.warning(f"Could not detect temporal frequency: {e}")
            return "unknown"
