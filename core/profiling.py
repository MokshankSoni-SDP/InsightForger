"""
Semantic profiling module.

Detects entities, relationships, and semantic types in the dataset.
Enhanced with cardinality intelligence, distribution awareness, and confidence scoring.
"""
import polars as pl
from typing import List, Dict, Any, Optional
from utils.schemas import EntityProfile, SemanticProfile, CandidateRelationship
from utils.helpers import (
    infer_statistical_type,
    infer_semantic_role,
    infer_distribution_type,
    calculate_basic_stats,
    get_logger
)

logger = get_logger(__name__)


class SemanticProfiler:
    """Semantic and statistical profiling of datasets with intelligence."""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.profile: Optional[SemanticProfile] = None
    
    def profile_dataset(self) -> SemanticProfile:
        """
        Perform comprehensive semantic profiling with intelligence.
        
        Returns:
            SemanticProfile with detected entities, metadata, and warnings
        """
        logger.info("Starting enhanced semantic profiling")
        
        entities = []
        time_columns = []
        numeric_columns = []
        categorical_columns = []
        identifier_columns = []
        profiling_warnings = []
        
        row_count = len(self.df)
        
        for col in self.df.columns:
            series = self.df[col]
            
            # Calculate cardinality metrics
            unique_count = series.n_unique()
            unique_ratio = unique_count / row_count if row_count > 0 else 0
            
            # Infer statistical type (fact-based)
            statistical_type = infer_statistical_type(series, unique_ratio)
            
            # Infer semantic role with confidence (interpretation-based)
            # Pass dataframe for neighbor validation
            semantic_guess, confidence = infer_semantic_role(series, unique_ratio, statistical_type, df=self.df)
            
            # Check if likely an identifier
            is_identifier = statistical_type == "identifier"
            
            # Infer distribution shape for numeric columns with zero validation
            # Pass dataframe and semantic role for intelligent zero validation
            distribution_type, sparsity_validated, sparsity_reason = infer_distribution_type(
                series, statistical_type, df=self.df, semantic_role=semantic_guess
            )
            
            # Calculate statistics
            stats = calculate_basic_stats(series)
            
            # Get sample values (non-null)
            sample_values = series.drop_nulls().head(5).to_list()
            
            # Track column categorization
            if statistical_type == "temporal":
                time_columns.append(col)
            elif statistical_type == "identifier":
                identifier_columns.append(col)
            elif statistical_type in ["numeric", "boolean"]:
                numeric_columns.append(col)
            elif statistical_type in ["categorical"]:
                categorical_columns.append(col)
            
            # Check if ordinal rank (for categorical that was reclassified from numeric)
            is_ordinal = False
            if statistical_type == "categorical" and "int" in str(series.dtype).lower():
                from utils.helpers import detect_ordinal_rank
                is_ordinal, _ = detect_ordinal_rank(series)
            
            # Create enhanced entity profile with intelligent metadata
            entity = EntityProfile(
                # Core identification
                column_name=col,
                data_type=str(series.dtype),
                
                # Statistical characteristics
                statistical_type=statistical_type,
                semantic_guess=semantic_guess,
                confidence=confidence,
                
                # Legacy compatibility (use semantic_guess for new code)
                entity_type=semantic_guess,
                
                # Cardinality intelligence
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                is_identifier=is_identifier,
                
                # Intelligent profiling enhancements
                is_ordinal=is_ordinal,
                sparsity_validated=sparsity_validated,
                sparsity_reason=sparsity_reason if sparsity_reason else None,
                
                # Distribution awareness
                distribution_type=distribution_type,
                
                # Sample data
                sample_values=sample_values,
                null_percentage=stats["null_percentage"]
            )
            entities.append(entity)
            
            logger.info(
                f"Profiled '{col}': stat_type={statistical_type}, "
                f"semantic={semantic_guess} (conf={confidence:.2f}), "
                f"dist={distribution_type}"
            )
            
            # Generate warnings for low confidence columns
            if confidence < 0.5:
                profiling_warnings.append(
                    f"Low confidence ({confidence:.2f}) for column '{col}' semantic type '{semantic_guess}'"
                )
        
        # Perform profiling completeness checks
        completeness_warnings = self._check_profiling_completeness(
            entities, time_columns, numeric_columns, identifier_columns
        )
        profiling_warnings.extend(completeness_warnings)
        
        # Detect candidate relationships with confidence and reasoning
        candidate_relationships = self._detect_candidate_relationships(
            entities, time_columns, numeric_columns, categorical_columns
        )
        
        # Create enhanced semantic profile
        self.profile = SemanticProfile(
            entities=entities,
            row_count=row_count,
            column_count=len(self.df.columns),
            
            # Column categorization
            time_columns=time_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            identifier_columns=identifier_columns,
            
            # Candidate relationships
            candidate_relationships=candidate_relationships,
            
            # Profiling metadata
            profiling_warnings=profiling_warnings
        )
        
        logger.info(
            f"Profiling complete: {len(entities)} entities, "
            f"{len(time_columns)} time, {len(numeric_columns)} numeric, "
            f"{len(identifier_columns)} identifiers, "
            f"{len(candidate_relationships)} relationships, "
            f"{len(profiling_warnings)} warnings"
        )
        
        return self.profile
    
    def _check_profiling_completeness(
        self,
        entities: List[EntityProfile],
        time_columns: List[str],
        numeric_columns: List[str],
        identifier_columns: List[str]
    ) -> List[str]:
        """
        Perform sanity checks on profiling completeness.
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for at least one numeric metric
        if not numeric_columns:
            warnings.append("No numeric columns detected - analysis capabilities limited")
        
        # Warn if multiple time columns (should usually have one primary)
        if len(time_columns) > 1:
            warnings.append(
                f"Multiple time columns detected ({len(time_columns)}): {', '.join(time_columns[:3])}... "
                "Consider specifying primary time column for time series analysis"
            )
        
        # Check for all identifiers (dataset might be lookup table)
        if len(identifier_columns) == len(entities):
            warnings.append(
                "All columns appear to be identifiers - dataset may be a lookup table "
                "rather than analytical data"
            )
        
        # Check for sparse numeric columns
        sparse_columns = [
            e.column_name for e in entities
            if e.distribution_type == "sparse" and e.statistical_type == "numeric"
        ]
        if sparse_columns:
            warnings.append(
                f"Sparse numeric columns detected (>70% zeros): {', '.join(sparse_columns[:5])}"
            )
        
        # Check for high-cardinality categoricals (might be hidden identifiers)
        high_card_cats = [
            e.column_name for e in entities
            if e.statistical_type == "categorical" and e.unique_ratio > 0.8
        ]
        if high_card_cats:
            warnings.append(
                f"High-cardinality categorical columns (might be IDs): {', '.join(high_card_cats[:5])}"
            )
        
        return warnings
    
    def _detect_candidate_relationships(
        self,
        entities: List[EntityProfile],
        time_columns: List[str],
        numeric_columns: List[str],
        categorical_columns: List[str]
    ) -> List[CandidateRelationship]:
        """
        Detect candidate relationships with confidence and reasoning.
        
        Note: These are CANDIDATES, not confirmed relationships.
        Downstream agents should verify evidence.
        
        Returns:
            List of CandidateRelationship objects
        """
        relationships = []
        
        # Time series candidate
        if time_columns and numeric_columns:
            confidence = 0.7 if len(time_columns) == 1 else 0.5  # Lower if multiple time cols
            
            relationships.append(CandidateRelationship(
                type="time_series",
                description=f"{len(numeric_columns)} metrics tracked over time",
                confidence=confidence,
                reason=f"Detected {len(time_columns)} time column(s) and {len(numeric_columns)} numeric metrics",
                involved_columns={
                    "time": time_columns,
                    "metrics": numeric_columns[:10]  # Limit for brevity
                }
            ))
        
        # Product-financial candidate
        product_cols = [e.column_name for e in entities if e.semantic_guess == "product"]
        financial_cols = [e.column_name for e in entities if e.semantic_guess == "financial"]
        
        if product_cols and financial_cols:
            # Confidence based on semantic confidence of involved columns
            avg_confidence = sum(
                e.confidence for e in entities
                if e.column_name in product_cols or e.column_name in financial_cols
            ) / (len(product_cols) + len(financial_cols))
            
            relationships.append(CandidateRelationship(
                type="product_financial",
                description="Product performance can be analyzed financially",
                confidence=avg_confidence * 0.6,  # Relationship confidence < column confidence
                reason=f"Product entity present + financial metrics present",
                involved_columns={
                    "products": product_cols,
                    "financials": financial_cols
                }
            ))
        
        # Customer-financial candidate
        customer_cols = [e.column_name for e in entities if e.semantic_guess == "customer"]
        
        if customer_cols and financial_cols:
            avg_confidence = sum(
                e.confidence for e in entities
                if e.column_name in customer_cols or e.column_name in financial_cols
            ) / (len(customer_cols) + len(financial_cols))
            
            relationships.append(CandidateRelationship(
                type="customer_financial",
                description="Customer behavior can be analyzed financially",
                confidence=avg_confidence * 0.6,
                reason="Customer entity present + financial metrics present",
                involved_columns={
                    "customers": customer_cols,
                    "financials": financial_cols
                }
            ))
        
        # Inventory-product candidate
        inventory_cols = [e.column_name for e in entities if e.semantic_guess == "inventory"]
        
        if inventory_cols and product_cols:
            avg_confidence = sum(
                e.confidence for e in entities
                if e.column_name in inventory_cols or e.column_name in product_cols
            ) / (len(inventory_cols) + len(product_cols))
            
            relationships.append(CandidateRelationship(
                type="product_inventory",
                description="Product inventory levels tracked",
                confidence=avg_confidence * 0.65,
                reason="Product entity + inventory metrics present",
                involved_columns={
                    "products": product_cols,
                    "inventory": inventory_cols
                }
            ))
        
        logger.info(f"Detected {len(relationships)} candidate relationships")
        return relationships
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get human-readable profile summary."""
        if self.profile is None:
            return {}
        
        # Count by semantic type
        semantic_counts = {}
        for entity in self.profile.entities:
            semantic_type = entity.semantic_guess
            semantic_counts[semantic_type] = semantic_counts.get(semantic_type, 0) + 1
        
        # Count by statistical type
        statistical_counts = {}
        for entity in self.profile.entities:
            stat_type = entity.statistical_type
            statistical_counts[stat_type] = statistical_counts.get(stat_type, 0) + 1
        
        return {
            "total_rows": self.profile.row_count,
            "total_columns": self.profile.column_count,
            "semantic_breakdown": semantic_counts,
            "statistical_breakdown": statistical_counts,
            "time_columns": self.profile.time_columns,
            "numeric_columns": self.profile.numeric_columns,
            "categorical_columns": self.profile.categorical_columns,
            "identifier_columns": self.profile.identifier_columns,
            "candidate_relationships": len(self.profile.candidate_relationships),
            "warnings": len(self.profile.profiling_warnings)
        }
    
    def detect_relationships(self) -> List[CandidateRelationship]:
        """
        Get candidate relationships from profile.
        
        Returns:
            List of CandidateRelationship objects
        """
        if self.profile is None:
            return []
        
        return self.profile.candidate_relationships
