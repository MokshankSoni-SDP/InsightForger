"""
Metric Blueprint Module.

Responsible for classifying abstract business intents into concrete metric types
WITHOUT referencing specific dataset columns. This serves as the schematic
for the MetricResolver.
"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from utils.helpers import get_logger

logger = get_logger(__name__)

class MetricType(str, Enum):
    DIRECT = "direct"               # Simple column Look-up (e.g. "Revenue")
    RATIO = "ratio"                 # A / B (e.g. "Cost per Sale")
    RATE = "rate"                   # A / Total (e.g. "Conversion Rate")
    EFFICIENCY = "efficiency"       # Input / Output or vice versa
    FUNNEL = "funnel_metric"        # Sequential steps
    INVENTORY = "inventory_metric"  # Stock vs Time/Sales
    TEMPORAL = "temporal_metric"    # Growth rates, YoY
    UNKNOWN = "unknown"

class MetricBlueprint(BaseModel):
    """
    Abstract definition of a metric's structure.
    Does NOT contain column names.
    """
    metric_type: MetricType
    primary_concept: str = Field(description="The main concept (numerator if ratio)")
    secondary_concept: Optional[str] = Field(default=None, description="The secondary concept (denominator if ratio)")
    required_dimensions: List[str] = Field(default_factory=list, description="Time, Geo, etc.")
    
    # Classification rationale (for transparency)
    rationale: str = ""

class BlueprintClassifier:
    """
    Classifies a business_metric string into a MetricBlueprint.
    Uses heuristic rules + lightweight LLM (optional, using rule-based for now for determinism).
    """
    
    @staticmethod
    def classify(business_metric: str, required_roles: List[str]) -> MetricBlueprint:
        """
        Classify metric based on name and roles.
        Deterministic rule-based approach for Phase 5.
        """
        metric_lower = business_metric.lower()
        roles = set(required_roles)
        
        # 1. Ratios / Rates / Efficiency
        if any(x in metric_lower for x in [" per ", "/", "ratio", "efficiency", "return on", "roi", "roas"]):
            # Generic ratio default
            m_type = MetricType.RATIO
            
            if "efficiency" in metric_lower or "roi" in metric_lower or "roas" in metric_lower:
                m_type = MetricType.EFFICIENCY
            
            # Smart split attempt
            parts = metric_lower.split(" per ")
            if len(parts) == 2:
                return MetricBlueprint(
                    metric_type=m_type,
                    primary_concept=parts[0].strip(),
                    secondary_concept=parts[1].strip(),
                    rationale=f"Detected '{m_type.value}' pattern via 'per' keyword"
                )
            elif "return on" in metric_lower:
                 return MetricBlueprint(
                    metric_type=m_type,
                    primary_concept="return",
                    secondary_concept=metric_lower.replace("return on", "").strip(),
                    rationale="Detected Return On X pattern"
                )
            
            return MetricBlueprint(
                metric_type=m_type,
                primary_concept=business_metric,
                secondary_concept="context_dependent",
                rationale="Detected ratio/efficiency keyword"
            )

        # 2. Rates (Percentage)
        if any(x in metric_lower for x in ["rate", "percent", "%", "share of"]):
             return MetricBlueprint(
                metric_type=MetricType.RATE,
                primary_concept=business_metric,
                rationale="Detected 'rate' or percentage keyword"
            )
            
        # 3. Funnel
        if "funnel" in roles or any(x in metric_lower for x in ["conversion", "drop off", "funnel"]):
            return MetricBlueprint(
                metric_type=MetricType.FUNNEL,
                primary_concept=business_metric,
                rationale="Detected funnel semantic role or keyword"
            )
            
        # 4. Inventory
        if "inventory" in roles and any(x in metric_lower for x in ["turnover", "days", "stock", "fill rate"]):
             return MetricBlueprint(
                metric_type=MetricType.INVENTORY,
                primary_concept=business_metric,
                required_dimensions=["time"],
                rationale="Detected inventory semantic role + standard metrics"
            )

        # 5. Temporal (Growth, YoY)
        if any(x in metric_lower for x in ["growth", "yoy", "mom", "change"]):
             return MetricBlueprint(
                metric_type=MetricType.TEMPORAL,
                primary_concept=business_metric,
                required_dimensions=["time"],
                rationale="Detected temporal change keywords"
            )

        # 6. Default: Direct
        return MetricBlueprint(
            metric_type=MetricType.DIRECT,
            primary_concept=business_metric,
            rationale="No complex patterns detected; assuming direct metric lookup"
        )
