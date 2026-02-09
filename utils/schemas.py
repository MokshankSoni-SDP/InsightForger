"""
Pydantic schemas for type validation and structured data across the pipeline.
"""
from typing import List, Dict, Any, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# ANALYTICAL BLUEPRINT SCHEMAS (Phase 0.5 Enhancement)
# ============================================================================

class MeasureDefinition(BaseModel):
    """Measure with aggregation semantics for execution planning."""
    column: str = Field(description="Column name")
    aggregation_type: Literal["SUM", "MEAN", "LAST", "MAX", "MIN", "DISTINCT_COUNT"] = Field(
        description="How to aggregate this measure"
    )
    rationale: str = Field(description="Why this aggregation method (for debugging)")


class KPIDefinition(BaseModel):
    """KPI with executable formula components for metric resolution."""
    name: str = Field(description="KPI name")
    kpi_type: str = Field(description="KPI type from KPI_TYPE_REQUIREMENTS")
    formula_template: str = Field(description="Formula from FORMULA_LIBRARY")
    
    # Component mapping (Flaw E fix)
    numerator: Optional[str] = Field(default=None, description="Column name for numerator")
    numerator_agg: Optional[str] = Field(default=None, description="Aggregation for numerator (SUM, MEAN, etc.)")
    denominator: Optional[str] = Field(default=None, description="Column name for denominator")
    denominator_agg: Optional[str] = Field(default=None, description="Aggregation for denominator")
    
    supporting_columns: List[str] = Field(default_factory=list, description="Columns used in this KPI")
    executable_formula: str = Field(default="", description="Pseudo-SQL formula for Phase 3")


class TemporalBehavior(BaseModel):
    """Time-based analysis guidance for insight depth."""
    primary_periodicity: Literal["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"] = Field(
        description="Primary time window for analysis"
    )
    rationale: str = Field(description="Why this periodicity fits the business")
    critical_slices: List[str] = Field(
        default_factory=list,
        description="Time-based segments (e.g., 'Is_Weekend', 'Is_Month_Start')"
    )
    seasonality_expected: bool = Field(description="True if seasonal patterns likely")

class LensRecommendation(BaseModel):
    """Analytical lens recommendation for Council of Lenses (Phase 1)."""
    lens_name: str = Field(description="Name of the analytical lens (e.g., 'Marketing Efficiency')")
    priority: int = Field(ge=1, description="Priority ranking (1 = highest)")
    objective: str = Field(description="1-sentence analytical goal for this lens")
    supporting_columns: List[str] = Field(description="Column names that justify this lens")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this lens recommendation")


class InsightDomain(BaseModel):
    """Opportunity Surface: High-impact dimension-metric pair identified in Phase 0.5."""
    domain_name: str = Field(description="Human-readable domain name (e.g., 'Brand-level Revenue Concentration')")
    primary_dimension: str = Field(description="Column name of the dimension")
    primary_metric: str = Field(description="Column name of the metric")
    why_valuable: str = Field(description="Business rationale for this domain")
    top_values_seen: List[str] = Field(
        description="Actual unique values from data (prevents hallucination)"
    )
    variance_score: float = Field(
        ge=0.0, le=1.0, 
        description="0-1 score indicating variance/concentration in this domain"
    )



class EAMMapping(BaseModel):
    """Entity-Attribute-Measure functional mapping for metric resolution."""
    anchor: str = Field(description="Primary entity name (e.g., 'Product', 'Customer')")
    anchor_column: str = Field(description="Column representing the anchor entity")
    attributes: List[str] = Field(description="Dimension columns (how we slice the data)")
    measures: List[MeasureDefinition] = Field(description="Fact columns with aggregation semantics")
    relationships: Dict[str, str] = Field(default_factory=dict, description="How attributes connect to anchors")
    cardinality: Dict[str, str] = Field(default_factory=dict, description="Relationship cardinality (1:1 or 1:Many)")


class DomainGuardrail(BaseModel):
    """Domain-specific rule or data quality check for execution phase."""
    rule_type: Literal["inverted_rank", "data_quality", "business_logic"] = Field(
        description="Type of guardrail rule"
    )
    column: str = Field(description="Column affected by this rule")
    description: str = Field(description="Human-readable explanation of the rule")
    validation_logic: str = Field(description="Pseudo-code or constraint definition")
    
    # Executable flags (Flaw D fix)
    invert_axis: bool = Field(default=False, description="True if lower values are better (for plotting)")
    filter_condition: Optional[str] = Field(default=None, description="SQL-like filter if needed")
    transformation: Optional[str] = Field(default=None, description="LOG, SQRT, etc. if needed")


# ============================================================================
# BUSINESS CONTEXT (Enhanced with Analytical Blueprint)
# ============================================================================

class BusinessContext(BaseModel):
    """Inferred business context from LLM with confidence and analytical blueprint."""
    # Core context
    industry: str = Field(description="Most likely industry")
    business_type: str = Field(description="Type of business operation")
    top_kpis: List[KPIDefinition] = Field(description="Top KPIs with executable formulas")
    additional_context: Optional[str] = Field(default=None, description="Additional insights")
    
    # Confidence and reasoning
    confidence: float = Field(ge=0.0, le=1.0, description="LLM confidence in this context inference")
    internal_reasoning: str = Field(description="INTERNAL ONLY: LLM reasoning - never expose in reports/UI")
    
    # Analytical Blueprint (Phase 0.5 Enhancement)
    recommended_lenses: List[LensRecommendation] = Field(
        default_factory=list,
        description="Analytical lenses to activate in Phase 1 (Council of Lenses)"
    )
    insight_domains: List[InsightDomain] = Field(
        default_factory=list,
        description="Opportunity Surfaces (high-impact dimension-metric pairs) for Phase 1 expansion"
    )
    eam_mapping: List[EAMMapping] = Field(
        default_factory=list,
        description="Entity-Attribute-Measure functional relationships for Phase 2.5"
    )
    guardrails: List[DomainGuardrail] = Field(
        default_factory=list,
        description="Domain-specific rules and data quality checks for Phase 3"
    )
    
    # Temporal Dynamics (Flaw F fix)
    temporal_behavior: Optional[TemporalBehavior] = Field(
        default=None,
        description="Time-based analysis guidance for insight depth"
    )


class EntityProfile(BaseModel):
    """Detected entity in the dataset with intelligence metadata."""
    # Core identification
    column_name: str
    data_type: str
    
    # Statistical characteristics
    statistical_type: Literal["numeric", "categorical", "temporal", "identifier", "boolean", "other"]
    semantic_guess: Literal["time", "product", "financial", "inventory", "funnel", "customer", "identifier", "other"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in semantic classification")
    
    # Legacy support (deprecated but kept for compatibility)
    entity_type: Optional[str] = Field(default=None, description="DEPRECATED: Use semantic_guess instead")
    
    # Cardinality intelligence
    unique_count: int
    unique_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of unique values to total rows")
    is_identifier: bool = Field(description="True if likely an ID/key column")
    
    # Intelligent profiling enhancements
    is_ordinal: bool = Field(default=False, description="True if detected as ordinal rank/position")
    sparsity_validated: bool = Field(default=False, description="True if high zero ratio was validated as business reality")
    sparsity_reason: Optional[str] = Field(default=None, description="Explanation for sparsity validation")
    
    # Distribution awareness
    distribution_type: Literal["normal", "skewed", "long_tail", "sparse", "uniform", "unknown"] = "unknown"
    
    # Sample data
    sample_values: List[Any]
    null_percentage: float


class CandidateRelationship(BaseModel):
    """A potential relationship between entities with confidence."""
    type: str = Field(description="Type of relationship (e.g., time_series, product_financial)")
    description: str = Field(description="Human-readable description")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this relationship")
    reason: str = Field(description="Why this relationship was detected")
    involved_columns: Dict[str, List[str]] = Field(description="Columns involved by role")


class SemanticProfile(BaseModel):
    """Complete semantic profile of the dataset with intelligence."""
    entities: List[EntityProfile]
    row_count: int
    column_count: int
    
    # Column categorization
    time_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    identifier_columns: List[str] = Field(
        default_factory=list,
        description="DERIVED: Programmatically generated from EntityProfile.is_identifier"
    )
    
    # Candidate relationships with confidence
    candidate_relationships: List[CandidateRelationship] = Field(default_factory=list)
    
    # Profiling metadata
    profiling_warnings: List[str] = Field(default_factory=list, description="Warnings about data quality or completeness")
    
    # Data grain (added by grain discovery)
    grain: Optional[Any] = None  # GrainProfile from grain_discovery module
    
    # Business context (added by LLM)
    business_context: Optional[BusinessContext] = None



# ============================================================================
# PHASE 1 ENHANCEMENTS (Analytical Engineering)
# ============================================================================

class AggregationScope(str, Enum):
    """
    Aggregation scope for hypothesis execution (Flaw 2 fix).
    
    Prevents "total" hallucination by forcing explicit scope definition.
    """
    GLOBAL = "GLOBAL"          # Entire dataset (e.g., total revenue across all time)
    TEMPORAL = "TEMPORAL"      # Time-based aggregation (e.g., daily average, weekly sum)
    DIMENSIONAL = "DIMENSIONAL" # Category-based aggregation (e.g., per product, per region)



class Hypothesis(BaseModel):
    """
    A testable hypothesis generated by a lens (Refactored for Analytical Engineering).
    
    Enhancements:
    - Explicit aggregation_scope (Flaw 2 fix)
    - Numerator/denominator concepts (Flaw 2 fix)
    - Dimensions and time_grain (Flaw 5 fix)
    - Guardrail linkage (Flaw 3 fix)
    """
    id: str = Field(description="Unique identifier (format: {lens}_{timestamp}_{metric})")
    lens: str = Field(description="Lens name from Phase 0.5 (e.g., 'Marketing Efficiency')")
    title: str = Field(description="Short hypothesis title")
    
    # --- ANALYTICAL ENGINEERING FIELDS (Phase 1 Enhancement) ---
    business_metric: str = Field(description="Primary metric column name (not abstract concept)")
    
    # Flaw 2 Fix: Explicit aggregation scope
    aggregation_scope: AggregationScope = Field(description="How to aggregate: GLOBAL, TEMPORAL, or DIMENSIONAL")
    time_grain: Optional[str] = Field(default=None, description="Time granularity if TEMPORAL (e.g., 'daily', 'weekly')")
    
    # Motif Identification (Phase 0.5/1 Enhancement)
    motif: Optional[Literal["Pareto", "Elasticity", "Velocity", "Benchmark"]] = Field(
        default=None,
        description="Analytical motif used to generate this hypothesis (guides Phase 3 execution strategy)"
    )

    
    # Dimensions and metric structure
    dimensions: List[str] = Field(default_factory=list, description="Columns to slice by (e.g., ['category', 'region'])")
    metric_template: str = Field(description="Math type: SUM, RATIO, GROWTH, MEAN, etc.")
    numerator_concept: str = Field(description="Column name for numerator")
    denominator_concept: Optional[str] = Field(default=None, description="Column name for denominator (or None)")
    
    # Flaw G Fix: Denominator scope (prevents sales/sales=1 trap)
    denominator_scope: Optional[Literal["SAME_ROW", "GLOBAL_SUM", "GROUP_SUM"]] = Field(
        default=None,
        description="How to interpret denominator: SAME_ROW (different column), GLOBAL_SUM (sum of all), GROUP_SUM (sum per group)"
    )
    
    # Phase 2 Output Integration
    polars_expression: Optional[str] = Field(default=None, description="Pre-resolved Polars expression from Phase 2")
    
    # Flaw 3 Fix: Guardrail linkage
    guardrail_applied: Optional[str] = Field(default=None, description="Guardrail rule_type if applicable (e.g., 'inverted_rank')")
    
    # Flaw I Fix: Explicit guardrail transformation
    guardrail_transformation: Optional[str] = Field(
        default=None,
        description="Mathematical transformation for guardrail (e.g., 'INVERT(our_position) = (11 - our_position)')"
    )
    
    # Priority and confidence
    priority: int = Field(ge=1, le=10, description="Priority from Phase 0.5 (1 = highest)")
    confidence: float = Field(ge=0.0, le=1.0, description="LLM confidence in this hypothesis")
    
    # Legacy fields (kept for backward compatibility)
    description: str = Field(default="", description="Detailed description")
    required_semantic_roles: List[str] = Field(default_factory=list, description="Semantic roles required (legacy)")
    expected_insight_type: Literal[
        "correlation", "trend", "anomaly", "causal", "forecast",
        "segmentation", "cohort", "efficiency", "distribution", "seasonality", "outlier_impact"
    ] = Field(default="trend", description="Type of insight expected")
    
    # --- CONCRETE RESOLUTION (Code Generated) ---
    resolved_metric: Optional[str] = Field(default=None, description="Actual column name resolved from business_metric")
    resolution_confidence: float = Field(default=0.0, description="Confidence in mapping intent to column")
    resolution_type: Literal["direct", "derived", "failed", "none"] = Field(default="none", description="Type of resolution")
    formula_name: Optional[str] = Field(default=None, description="Name of derived metric formula if applicable")
    dependencies: List[str] = Field(default_factory=list, description="List of columns used for this metric")
    
    computation_plan: str = Field(default="", description="Python code plan to execute (populated after resolution)")

    # Intelligence metadata
    confidence: float = Field(ge=0.0, le=1.0, description="LLM-estimated confidence in the business value")
    priority: int = Field(ge=1, le=5, description="Priority level: 1 (low) to 5 (critical)")
    
    # Deprecated fields mapping (for backward compatibility during refactor, strictly optional)
    related_metric: Optional[str] = None



class ComputationResult(BaseModel):
    """Result from executing a hypothesis computation."""
    hypothesis_id: str
    success: bool
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    execution_time: float
    retry_count: int = 0


class ValidationResult(BaseModel):
    """Result from adversarial validation with verdict levels and explainability."""
    hypothesis_id: str
    verdict: Literal["accept", "weak_signal", "reject"] = Field(
        description="Verdict level: accept (high confidence), weak_signal (exploratory), reject (not trustworthy)"
    )
    rejection_reason: Optional[str] = None
    statistical_tests: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(ge=0.0, le=1.0)
    passed_checks: List[str] = Field(default_factory=list, description="List of validation checks that passed")
    failed_checks: List[str] = Field(default_factory=list, description="List of validation checks that failed")
    
    @property
    def passed(self) -> bool:
        """DERIVED: True if verdict is accept or weak_signal."""
        return self.verdict in ["accept", "weak_signal"]


class ValidatedInsight(BaseModel):
    """A validated insight ready for narration."""
    hypothesis_id: str
    lens: str
    title: str
    metric: str
    result_data: Dict[str, Any]
    validation: ValidationResult
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Tracking resolution details
    resolution_type: str = Field(default="direct")
    formula_name: Optional[str] = None


class NarratedInsight(BaseModel):
    """An insight with natural language narration and verdict level."""
    insight_id: str
    lens: str
    title: str
    key_finding: str = Field(description="What was discovered")
    why_it_matters: str = Field(description="Business significance")
    what_caused_it: Optional[str] = Field(default=None, description="Root cause if identifiable")
    recommendation: str = Field(description="Actionable next steps")
    supporting_data: Dict[str, Any]
    confidence: Literal["high", "medium", "low"] = Field(
        description="DERIVED from validation.confidence_score, NOT from LLM judgment"
    )
    verdict: Literal["accept", "weak_signal", "reject"] = Field(
        default="accept", description="Validation verdict: determines narration tone"
    )
    
    # Transparency
    resolution_type: str = Field(default="direct")
    formula_name: Optional[str] = None
    
    # UI/Reporting fields (populated after narration)
    chart_path: Optional[str] = None
    key_evidence: List[Dict[str, str]] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Overall execution plan from the planner."""
    hypotheses: List[Hypothesis]
    execution_order: List[str] = Field(description="Hypothesis IDs in execution order")
    estimated_duration: Optional[float] = None


class PipelineProgress(BaseModel):
    """Current progress of the pipeline."""
    phase: Literal["ingest", "context", "profiling", "hypotheses", "execution", "validation", "narration", "reporting", "complete"]
    progress_percentage: float = Field(ge=0.0, le=100.0)
    current_step: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Final analysis report structure."""
    dataset_name: str
    analysis_timestamp: datetime
    business_context: BusinessContext
    insights: List[NarratedInsight]
    charts: List[str] = Field(description="Paths to generated chart files")
    executive_summary: str
    total_hypotheses: int
    validated_insights: int
    rejection_count: int


# ============================================================================
# PHASE 2: EXECUTION PLAN (Autonomous Metric Resolution)
# ============================================================================

class Phase2Plan(BaseModel):
    """
    Validated execution plan for Phase 3.
    
    Contains all information needed to execute hypothesis without LLM.
    Generated by autonomous Phase 2 metric resolver.
    """
    hypothesis_id: str = Field(description="ID of hypothesis this plan executes")
    
    # Resolved columns (physical column names from dataset)
    numerator_column: str = Field(description="Physical column name for numerator")
    denominator_column: Optional[str] = Field(default=None, description="Physical column name for denominator (if ratio)")
    
    # Executable formula (ready for Polars)
    polars_expression: str = Field(description="Executable Polars expression")
    
    # Aggregation details
    aggregation_scope: AggregationScope = Field(description="How to aggregate")
    time_grain: Optional[str] = Field(default=None, description="Time granularity if TEMPORAL")
    dimensions: List[str] = Field(default_factory=list, description="Columns to group by")
    
    # Validation metadata
    type_safe: bool = Field(description="Both columns are numeric and validated")
    has_guardrails: bool = Field(default=False, description="Guardrails applied to formula")
    null_safe: bool = Field(default=True, description="Formula handles null values")
    
    # Resolution metadata (for debugging/transparency)
    resolution_method: str = Field(description="How columns were found: exact|semantic|fuzzy")
    resolution_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in column mapping")
    
    # Original hypothesis reference
    hypothesis_title: str = Field(description="Original hypothesis title for logging")
    lens_name: str = Field(description="Lens that generated this hypothesis")
