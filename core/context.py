"""
Phase 0.5: Context Injection with Analytical Blueprint

Uses LLM to infer business context from semantic profile and generate prescriptive
analytical blueprint including lens orchestration, EAM mapping with aggregation types,
KPI component mapping, temporal behavior, and domain guardrails.

Enhancements:
- Intelligent column selection (information density scoring)
- Semantic validation (KPI type requirements)
- Statistical fingerprinting for generic column names
- Analytical blueprint generation (lenses, EAM, guardrails)
- Aggregation type specification for measures
- KPI numerator/denominator mapping
- Temporal periodicity guidance
"""
import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq
from utils.schemas import (
    BusinessContext, SemanticProfile, EntityProfile,
    LensRecommendation, EAMMapping, DomainGuardrail,
    MeasureDefinition, KPIDefinition, TemporalBehavior
)
from utils.helpers import get_logger

load_dotenv()
logger = get_logger(__name__)


# ============================================================================
# KPI TYPE REQUIREMENTS (Semantic Requirement Validation)
# ============================================================================
KPI_TYPE_REQUIREMENTS = {
    # Financial KPIs
    "financial_basic": {
        "required_roles": ["financial"],
        "min_count": 1,
        "examples": ["Revenue", "Profit", "Sales", "Cost"]
    },
    
    # Efficiency KPIs (ratios)
    "efficiency": {
        "required_roles": ["financial"],
        "min_count": 2,  # Need at least 2 financial metrics for ratio
        "examples": ["ROI", "ROAS", "Profit Margin", "Cost per Sale"]
    },
    
    # Growth KPIs
    "growth": {
        "required_roles": ["financial", "time"],
        "min_count": 2,
        "examples": ["Revenue Growth", "MoM Growth", "YoY Growth"]
    },
    
    # Customer KPIs
    "customer_value": {
        "required_roles": ["customer", "financial"],
        "min_count": 2,
        "examples": ["CLV", "CAC", "ARPU"]
    },
    
    # Retention KPIs
    "retention": {
        "required_roles": ["customer", "time"],
        "min_count": 2,
        "examples": ["Churn Rate", "Retention Rate"]
    },
    
    # Marketing KPIs
    "funnel": {
        "required_roles": ["funnel"],
        "min_count": 1,
        "examples": ["Conversion Rate", "CTR", "Engagement Rate"]
    },
    
    # Operational KPIs
    "inventory": {
        "required_roles": ["inventory", "time"],
        "min_count": 2,
        "examples": ["Inventory Turnover", "Stock Days"]
    },
    
    # Product KPIs
    "product": {
        "required_roles": ["product"],
        "min_count": 1,
        "examples": ["Product Performance", "Category Sales"]
    }
}


# ============================================================================
# FORMULA LIBRARY (Phase 2.5 Integration)
# ============================================================================
FORMULA_LIBRARY = {
    "SUM": "Aggregate by summing a metric",
    "AVERAGE": "Aggregate by averaging a metric",
    "COUNT": "Count occurrences or events",
    "RATIO": "Divide metric A by metric B",
    "PERCENT_CHANGE": "Calculate (New - Old) / Old * 100",
    "RATE": "Calculate events per time period"
}


# ============================================================================
# LENS LIBRARY (Analytical Blueprint Enhancement)
# ============================================================================
LENS_LIBRARY = {
    "Marketing Efficiency": {
        "description": "Analyze ad spend, ROAS, CPC, conversion rates",
        "required_roles": ["financial", "funnel"],
        "typical_objectives": [
            "Find optimal ad spend allocation across channels",
            "Identify underperforming campaigns to cut",
            "Calculate true customer acquisition cost"
        ]
    },
    "Inventory Health": {
        "description": "Analyze stock levels, turnover, dead stock risk",
        "required_roles": ["inventory", "financial"],
        "typical_objectives": [
            "Identify slow-moving inventory to liquidate",
            "Prevent stockouts on high-margin items",
            "Calculate inventory carrying cost and optimize"
        ]
    },
    "Pricing Strategy": {
        "description": "Analyze price elasticity, margin optimization",
        "required_roles": ["financial", "product"],
        "typical_objectives": [
            "Find price-demand sweet spot for each category",
            "Identify margin erosion risks from discounting",
            "Optimize discount strategies by segment"
        ]
    },
    "Customer Behavior": {
        "description": "Analyze purchase patterns, retention, churn",
        "required_roles": ["customer", "time"],
        "typical_objectives": [
            "Segment customers by lifetime value",
            "Predict churn risk and prevent defection",
            "Identify cross-sell and upsell opportunities"
        ]
    },
    "Operational Performance": {
        "description": "Analyze fulfillment, logistics, process efficiency",
        "required_roles": ["time", "identifier"],
        "typical_objectives": [
            "Reduce order processing time and costs",
            "Optimize warehouse operations and layout",
            "Identify process bottlenecks and inefficiencies"
        ]
    },
    "Financial Health": {
        "description": "Analyze revenue, profit, cash flow trends",
        "required_roles": ["financial", "time"],
        "typical_objectives": [
            "Track revenue trends and forecast future",
            "Identify profit leakage points in the value chain",
            "Forecast cash flow and working capital needs"
        ]
    },
    "Product Performance": {
        "description": "Analyze product sales, returns, ratings, mix",
        "required_roles": ["product", "financial"],
        "typical_objectives": [
            "Identify star vs. dog products in portfolio",
            "Optimize product mix for maximum profitability",
            "Reduce return rates and improve quality"
        ]
    },
    "Regional Analysis": {
        "description": "Analyze geographic performance patterns",
        "required_roles": ["location", "financial"],
        "typical_objectives": [
            "Identify high-potential markets for expansion",
            "Optimize regional pricing and promotions",
            "Allocate marketing budget by regional ROI"
        ]
    }
}


class ContextInjector:
    """Uses LLM to infer business context with intelligent column selection and validation."""
    
    def __init__(self, token_tracker=None):
        # Use primary GROQ_API_KEY for Phase 0.5 (rate limit distribution)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.token_tracker = token_tracker
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set")
        
        self.client = Groq(api_key=self.api_key)
    
    def _calculate_lens_budget(self, profile: SemanticProfile) -> int:
        """
        Calculate how many lenses to recommend based on data complexity.
        
        Dynamic Scaling Rule: N = number of unique anchors + 2
        - Simple data (1 anchor): 3 lenses
        - Multi-entity (3 anchors): 5 lenses
        - Complex (5+ anchors): 7-9 lenses
        
        This prevents token bloat while maintaining analytical depth.
        
        Args:
            profile: Semantic profile
            
        Returns:
            Maximum number of lenses to recommend
        """
        # Estimate number of anchors from identifier columns
        # Heuristic: Each identifier column represents a potential entity/anchor
        identifier_count = len(profile.identifier_columns)
        
        # Apply formula: N = anchors + 2
        # Minimum of 1 anchor (even if no identifiers, there's always "the row")
        estimated_anchors = max(1, identifier_count)
        lens_budget = estimated_anchors + 2
        
        # Cap at reasonable maximum to prevent token explosion
        lens_budget = min(lens_budget, 9)
        
        logger.info(f"Lens budget calculated: {lens_budget} lenses (based on {estimated_anchors} anchors)")
        return lens_budget
    
    def infer_context(self, profile: SemanticProfile) -> BusinessContext:
        """
        Use LLM to infer business context from semantic profile.
        
        Enhancements:
        - Information density scoring for column selection
        - Statistical fingerprinting for generic column names
        - Semantic requirement validation for KPIs
        - Chain-of-thought prompting
        - Formula library integration
        
        Args:
            profile: Semantic profile of the dataset
            
        Returns:
            BusinessContext with confidence and reasoning
        """
        logger.info("Inferring business context using Groq LLM (enhanced with intelligent selection)")
        
        # FIX FOR FLAW A: Select top columns by information density (not first 20)
        selected_entities = self._select_top_columns(profile, limit=20)
        logger.info(f"Selected {len(selected_entities)} most analytically significant columns")
        
        # FIX FOR FLAW C: Prepare enhanced context with statistical fingerprinting
        column_info = []
        for entity in selected_entities:
            # Convert sample values to strings for JSON serialization
            samples = [str(v) if v is not None else None for v in entity.sample_values[:3]]
            
            info = {
                "column": entity.column_name,
                "statistical_type": entity.statistical_type,
                "semantic_guess": entity.semantic_guess,
                "confidence": round(entity.confidence, 2),
                "samples": samples,
                "distribution": entity.distribution_type if hasattr(entity, 'distribution_type') else "unknown",
                "null_pct": round(entity.null_percentage, 1),
                "unique_ratio": round(entity.unique_ratio, 2)
            }
            
            # Add intelligent profiling metadata
            if hasattr(entity, 'is_ordinal'):
                info["is_ordinal"] = entity.is_ordinal
            if hasattr(entity, 'sparsity_validated'):
                info["sparsity_validated"] = entity.sparsity_validated
            
            # Add statistical fingerprint for numeric columns
            if entity.statistical_type == "numeric" and entity.sample_values:
                try:
                    samples_numeric = [float(v) for v in entity.sample_values if v is not None]
                    if samples_numeric:
                        info["stats"] = {
                            "min": round(min(samples_numeric), 2),
                            "max": round(max(samples_numeric), 2),
                            "mean": round(sum(samples_numeric) / len(samples_numeric), 2)
                        }
                except:
                    pass
            
            column_info.append(info)
        
        # FIX FOR FLAW B & E: Enhanced prompt with CoT and formula library
        prompt = self._build_enhanced_prompt(profile, column_info, selected_entities)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert business analyst configuring an autonomous analytical engine. Provide structured, prescriptive blueprints."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000  # Increased for analytical blueprint (lenses + EAM + guardrails)
            )
            
            content = response.choices[0].message.content
            logger.info(f"LLM response received: {len(content)} chars")
            
            # Robust JSON parsing
            context_data = self._extract_json_robustly(content)
            
            if not context_data:
                raise ValueError("Failed to extract valid JSON from LLM response")
            
            
            # Extract fields
            industry = context_data.get("industry", "Unknown")
            business_type = context_data.get("business_type", "Unknown")
            business_model = context_data.get("business_model", "")
            llm_kpis = context_data.get("top_kpis", [])
            additional_context = context_data.get("additional_context")
            confidence = float(context_data.get("confidence", 0.5))
            reasoning = context_data.get("reasoning", "No reasoning provided")
            
            # Parse KPIs with component mapping (Flaw E fix)
            logger.info("Parsing KPIs with component mapping...")
            parsed_kpis = []
            for kpi in llm_kpis:
                try:
                    parsed_kpis.append(KPIDefinition(
                        name=kpi.get("name", ""),
                        kpi_type=kpi.get("kpi_type", ""),
                        formula_template=kpi.get("formula_template", ""),
                        numerator=kpi.get("numerator"),
                        numerator_agg=kpi.get("numerator_agg"),
                        denominator=kpi.get("denominator"),
                        denominator_agg=kpi.get("denominator_agg"),
                        supporting_columns=kpi.get("supporting_columns", []),
                        executable_formula=kpi.get("executable_formula", "")
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse KPI: {e}")
            
            logger.info(f"Parsed {len(parsed_kpis)} KPIs with executable formulas")
            
            # Validate KPIs have required columns
            validated_kpis = []
            all_columns = {e.column_name for e in profile.entities}
            for kpi in parsed_kpis:
                # Check if numerator exists
                if kpi.numerator and kpi.numerator not in all_columns:
                    logger.warning(f"KPI '{kpi.name}' references non-existent column '{kpi.numerator}'")
                    continue
                # Check if denominator exists (if specified)
                if kpi.denominator and kpi.denominator not in all_columns:
                    logger.warning(f"KPI '{kpi.name}' references non-existent column '{kpi.denominator}'")
                    continue
                validated_kpis.append(kpi)
            
            # Adjust confidence based on validation
            if len(validated_kpis) < len(parsed_kpis):
                kpi_rejection_count = len(parsed_kpis) - len(validated_kpis)
                confidence = confidence * (len(validated_kpis) / len(parsed_kpis)) if parsed_kpis else 0.3
                reasoning += f" | Note: {kpi_rejection_count} KPI(s) removed due to missing columns"
                logger.warning(f"Removed {kpi_rejection_count} KPIs with invalid column references")
            
            # Ensure at least some KPIs if we have metrics
            if not validated_kpis and profile.numeric_columns:
                # Create fallback KPIs
                validated_kpis = [
                    KPIDefinition(
                        name="Revenue Analysis",
                        kpi_type="financial",
                        formula_template="SUM",
                        supporting_columns=profile.numeric_columns[:1]
                    )
                ]
                confidence = 0.3
                reasoning += " | Fallback KPIs used due to validation failure"
            
            # ANALYTICAL BLUEPRINT PARSING
            logger.info("Parsing analytical blueprint fields...")
            
            # Parse lens recommendations
            lens_data = context_data.get("recommended_lenses", [])
            recommended_lenses = []
            for lens in lens_data:
                try:
                    recommended_lenses.append(LensRecommendation(
                        lens_name=lens.get("lens_name", ""),
                        priority=int(lens.get("priority", 999)),
                        objective=lens.get("objective", ""),
                        supporting_columns=lens.get("supporting_columns", []),
                        confidence=float(lens.get("confidence", 0.5))
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse lens recommendation: {e}")
            
            logger.info(f"Parsed {len(recommended_lenses)} lens recommendations")
            
            # Parse EAM mapping with measure definitions (Flaw D fix)
            eam_data = context_data.get("eam_mapping", [])
            eam_mapping = []
            for eam in eam_data:
                try:
                    # Parse measures with aggregation types
                    measures_data = eam.get("measures", [])
                    parsed_measures = []
                    for measure in measures_data:
                        if isinstance(measure, dict):
                            # New format with aggregation types
                            parsed_measures.append(MeasureDefinition(
                                column=measure.get("column", ""),
                                aggregation_type=measure.get("aggregation_type", "SUM"),
                                rationale=measure.get("rationale", "")
                            ))
                        else:
                            # Old format (string) - default to SUM
                            logger.warning(f"Measure '{measure}' missing aggregation type, defaulting to SUM")
                            parsed_measures.append(MeasureDefinition(
                                column=str(measure),
                                aggregation_type="SUM",
                                rationale="Default aggregation (no type specified)"
                            ))
                    
                    eam_mapping.append(EAMMapping(
                        anchor=eam.get("anchor", ""),
                        anchor_column=eam.get("anchor_column", ""),
                        attributes=eam.get("attributes", []),
                        measures=parsed_measures,
                        relationships=eam.get("relationships", {}),
                        cardinality=eam.get("cardinality", {})
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse EAM mapping: {e}")
            
            logger.info(f"Parsed {len(eam_mapping)} EAM mappings with aggregation types")
            
            # Parse temporal behavior (Flaw F fix)
            temporal_data = context_data.get("temporal_behavior")
            temporal_behavior = None
            if temporal_data:
                try:
                    temporal_behavior = TemporalBehavior(
                        primary_periodicity=temporal_data.get("primary_periodicity", "Weekly"),
                        rationale=temporal_data.get("rationale", ""),
                        critical_slices=temporal_data.get("critical_slices", []),
                        seasonality_expected=temporal_data.get("seasonality_expected", False)
                    )
                    logger.info(f"Parsed temporal behavior: {temporal_behavior.primary_periodicity}")
                except Exception as e:
                    logger.warning(f"Failed to parse temporal behavior: {e}")
            
            # Parse guardrails with executable flags
            guardrail_data = context_data.get("guardrails", [])
            guardrails = []
            for gr in guardrail_data:
                try:
                    guardrails.append(DomainGuardrail(
                        rule_type=gr.get("rule_type", "business_logic"),
                        column=gr.get("column", ""),
                        description=gr.get("description", ""),
                        validation_logic=gr.get("validation_logic", ""),
                        invert_axis=gr.get("invert_axis", False),
                        filter_condition=gr.get("filter_condition"),
                        transformation=gr.get("transformation")
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse guardrail: {e}")
            
            logger.info(f"Parsed {len(guardrails)} guardrails")
            
            # Low confidence handling
            if confidence < 0.3:
                logger.warning(f"Low context confidence ({confidence:.2f}) - treating as generic business")
                industry = "General Business"
                business_type = "Data Analysis"
            
            business_context = BusinessContext(
                industry=industry,
                business_type=business_type,
                top_kpis=validated_kpis,
                additional_context=f"{business_model}. {additional_context}" if business_model else additional_context,
                confidence=confidence,
                internal_reasoning=reasoning,
                # Analytical Blueprint
                recommended_lenses=recommended_lenses,
                eam_mapping=eam_mapping,
                guardrails=guardrails,
                temporal_behavior=temporal_behavior
            )
            
            logger.info(f"Context inferred: {business_context.industry} - {business_context.business_type} (confidence: {confidence:.2f})")
            logger.info(f"Validated KPIs: {len(validated_kpis)} KPIs with executable formulas")
            logger.info(f"Analytical Blueprint: {len(recommended_lenses)} lenses, {len(eam_mapping)} EAM mappings, {len(guardrails)} guardrails")
            if temporal_behavior:
                logger.info(f"Temporal Behavior: {temporal_behavior.primary_periodicity} periodicity")
            
            return business_context
            
        except Exception as e:
            logger.error(f"Failed to infer context from LLM: {e}")
            return BusinessContext(
                industry="General Business",
                business_type="Data Analysis",
                top_kpis=[
                    KPIDefinition(
                        name="Revenue Analysis",
                        kpi_type="financial",
                        formula_template="SUM"
                    )
                ],
                additional_context="Unable to infer specific context due to error",
                confidence=0.2,
                internal_reasoning=f"Fallback context due to error: {str(e)}",
                recommended_lenses=[],
                eam_mapping=[],
                guardrails=[],
                temporal_behavior=None
            )
    
    def _calculate_information_density(self, entity: EntityProfile) -> float:
        """
        Calculate analytical importance score for column selection.
        
        Higher scores = more analytically significant columns.
        
        Scoring factors:
        - Statistical type (temporal > identifier > numeric > categorical)
        - Semantic confidence
        - Semantic role importance (financial, funnel, customer)
        - Data quality (penalize sparse, low-variance)
        - Cardinality (boost high-cardinality dimensions)
        
        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.0
        
        # Base score by statistical type
        if entity.statistical_type == "temporal":
            score = 1.0  # Essential for time-series context
        elif entity.statistical_type == "identifier":
            score = 0.9  # Essential for grain
        elif entity.statistical_type == "numeric":
            score = 0.6  # Likely a metric
        elif entity.statistical_type == "categorical":
            score = 0.5  # Likely a dimension
        else:
            score = 0.3  # Other types
        
        # Boost for high-confidence semantic roles
        if entity.confidence >= 0.80:
            score += 0.2
        elif entity.confidence >= 0.60:
            score += 0.1
        
        # Boost for important semantic roles
        if entity.semantic_guess in ["financial", "funnel", "customer"]:
            score += 0.15
        elif entity.semantic_guess in ["inventory", "product"]:
            score += 0.10
        
        # Penalty for sparse data (high null percentage)
        if entity.null_percentage > 70:
            score *= 0.3
        elif entity.null_percentage > 50:
            score *= 0.6
        
        # Penalty for low variance (all same value)
        if entity.unique_ratio < 0.01:
            score *= 0.2
        
        # Boost for high cardinality (likely key dimension)
        if entity.unique_ratio > 0.80:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _select_top_columns(self, profile: SemanticProfile, limit: int = 20) -> List[EntityProfile]:
        """
        Select most analytically significant columns using information density scoring.
        
        FIX FOR FLAW A: Instead of taking first 20 columns, rank by importance.
        
        Args:
            profile: Semantic profile
            limit: Maximum number of columns to select
            
        Returns:
            List of top N most important entities
        """
        # Score all entities
        scored_entities = [
            (entity, self._calculate_information_density(entity))
            for entity in profile.entities
        ]
        
        # Sort by score descending
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        
        # Log top scores for debugging
        logger.debug("Top column scores:")
        for entity, score in scored_entities[:5]:
            logger.debug(f"  {entity.column_name}: {score:.2f} ({entity.semantic_guess})")
        
        # Return top N entities
        return [entity for entity, score in scored_entities[:limit]]
    
    def _build_enhanced_prompt(
        self, profile: SemanticProfile, column_info: List[Dict], 
        selected_entities: List[EntityProfile]
    ) -> str:
        """
        Build enhanced prompt with Analytical Blueprint instructions.
        
        Transforms LLM from descriptive to prescriptive mode:
        - Lens orchestration (which analytical angles to activate)
        - EAM mapping (functional relationships for metric resolution)
        - Guardrails (domain-specific rules for execution)
        """
        # Calculate lens budget for dynamic scaling
        lens_budget = self._calculate_lens_budget(profile)
        
        prompt = f"""You are NOT just describing the data. You are CONFIGURING the next 5 agents in the analytical factory.

=== YOUR ROLE ===
Your output is a BLUEPRINT that tells:
- Phase 1 (Council of Lenses): Which analytical angles to activate
- Phase 2.5 (Metric Resolution): How columns relate functionally  
- Phase 3 (Execution): What domain rules to enforce

=== DATASET SUMMARY ===
- Total Rows: {profile.row_count:,}
- Total Columns: {profile.column_count}
- Columns Analyzed: {len(selected_entities)} (top ranked by analytical significance)
- Time Columns: {', '.join(profile.time_columns[:5]) if profile.time_columns else 'None'}
- Numeric Columns: {len(profile.numeric_columns)}
- Identifier Columns: {', '.join(profile.identifier_columns[:3]) if profile.identifier_columns else 'None'}

=== COLUMN DETAILS (with statistical fingerprints) ===
{json.dumps(column_info, indent=2)}

=== AVAILABLE ANALYTICAL LENSES ===
{json.dumps({k: {"description": v["description"], "required_roles": v["required_roles"]} for k, v in LENS_LIBRARY.items()}, indent=2)}

=== AVAILABLE KPI TYPES ===
{json.dumps({k: v['examples'] for k, v in KPI_TYPE_REQUIREMENTS.items()}, indent=2)}

=== AVAILABLE FORMULA TEMPLATES ===
{json.dumps(FORMULA_LIBRARY, indent=2)}

=== INSTRUCTIONS ===

STEP 1: ANALYZE
Look at the column names, samples, and statistical fingerprints.
Write a 3-sentence deduction:
1. Identify the main entities (Product, Customer, Transaction, Order, etc.)
2. Identify the primary revenue/success metric (Sales, Revenue, Conversions, etc.)
3. Identify the relationship between Marketing and Sales (if applicable)

STEP 2: DEDUCE
If you were the CEO, which 3 columns keep you awake at night?
If you had to write 5 SQL queries to find "hidden money" in this data, what would the WHERE and GROUP BY clauses be?

STEP 3: ORCHESTRATE LENSES (CRITICAL)
You MUST recommend exactly {lens_budget} lenses (no more, no less).

Instructions:
1. Review the AVAILABLE ANALYTICAL LENSES above
2. Select the top {lens_budget} lenses ranked by "Impact Potential" for THIS specific dataset
3. ONLY include a lens if the supporting columns have semantic confidence > 0.70
4. For each lens, provide:
   - lens_name: Exact name from the library
   - priority: 1 = highest impact
   - objective: 1 sentence starting with "Analyze X to find/identify/optimize Y"
   - supporting_columns: Actual column names from the dataset
   - confidence: 0.0-1.0 based on data support

Example lens recommendation:
{{
  "lens_name": "Marketing Efficiency",
  "priority": 1,
  "objective": "Analyze ad spend vs sales across positions to find optimal CPC sweet spot",
  "supporting_columns": ["ad_cost", "sales", "position"],
  "confidence": 0.90
}}

STEP 4: MAP FUNCTIONAL RELATIONSHIPS (EAM) - ENHANCED
For each "Anchor" (entity the row is about):
- anchor: Entity name (e.g., "Product", "Customer", "Transaction")
- anchor_column: Column representing it (e.g., "product_id")
- attributes: Dimension columns (how we slice) - categories, segments, ranks
- measures: List of measure objects with aggregation semantics
- relationships: How attributes connect to anchors
- cardinality: Relationship cardinality (1:1 or 1:Many)

FOR EACH MEASURE, YOU MUST SPECIFY:
- column: Column name
- aggregation_type: Choose from [SUM, MEAN, LAST, MAX, MIN, DISTINCT_COUNT]
- rationale: Why this aggregation (for debugging)

AGGREGATION RULES (CRITICAL):
- Financial metrics (sales, revenue, cost, price): SUM
- Ranks/Positions: MEAN (NEVER SUM - summing ranks is meaningless)
- Snapshots (inventory, stock, balance): LAST or MEAN (NOT SUM over time)
- Counts (orders, users, transactions): DISTINCT_COUNT
- Rates/Percentages: MEAN
- Timestamps: MAX or MIN (for first/last occurrence)

CARDINALITY RULES:
- "1:1": Product → Brand (one product has one brand)
- "1:Many": Category → Product (one category has many products)

Example EAM mapping:
{{
  "anchor": "Product",
  "anchor_column": "productid",
  "attributes": ["category", "subcategory", "brand"],
  "measures": [
    {{
      "column": "sales",
      "aggregation_type": "SUM",
      "rationale": "Revenue should be summed across products"
    }},
    {{
      "column": "our_position",
      "aggregation_type": "MEAN",
      "rationale": "Position is a rank, averaging shows typical placement"
    }},
    {{
      "column": "inventory",
      "aggregation_type": "LAST",
      "rationale": "Inventory is a snapshot, we want current stock not sum over time"
    }}
  ],
  "relationships": {{"category": "Product", "subcategory": "Product"}},
  "cardinality": {{"category": "1:Many", "subcategory": "1:Many"}}
}}

STEP 5: DEFINE KPI COMPONENTS (CRITICAL - Flaw E Fix)
For EVERY KPI you suggest, you MUST provide executable formula components.

REQUIRED FIELDS:
- name: KPI name
- kpi_type: From KPI types
- formula_template: From formula library
- numerator: Actual column name (use snake_case from dataset)
- numerator_agg: How to aggregate numerator (SUM, MEAN, etc.)
- denominator: Actual column name (or null if COUNT-based)
- denominator_agg: How to aggregate denominator
- supporting_columns: List of columns used
- executable_formula: Pseudo-SQL showing exact calculation

CRITICAL: If a KPI requires a column that doesn't exist, DO NOT suggest it.
Example: "Average Order Value" requires order_id. If no order_id exists, suggest "Average Product Value" using productid instead.

Example KPI:
{{
  "name": "Return on Ad Spend (ROAS)",
  "kpi_type": "efficiency",
  "formula_template": "RATIO",
  "numerator": "sales",
  "numerator_agg": "SUM",
  "denominator": "adcost",
  "denominator_agg": "SUM",
  "supporting_columns": ["sales", "adcost"],
  "executable_formula": "SUM(sales) / SUM(adcost)"
}}

STEP 6: DEFINE TEMPORAL BEHAVIOR (Flaw F Fix)
Based on the industry and data grain, specify:
- primary_periodicity: Daily | Weekly | Monthly | Quarterly | Yearly
- rationale: Why this periodicity fits the business
- critical_slices: Time-based segments (e.g., ["Is_Weekend", "Is_Month_End"])
- seasonality_expected: true | false

PERIODICITY RULES:
- Retail/E-commerce: Weekly (weekend peaks)
- SaaS/Subscriptions: Monthly (billing cycles)
- B2B: Quarterly (sales cycles)
- Advertising: Daily (campaign optimization)
- Manufacturing: Monthly (production cycles)

Example:
{{
  "primary_periodicity": "Weekly",
  "rationale": "Retail sales peak on weekends, weekly view captures this pattern",
  "critical_slices": ["Is_Weekend", "Is_Month_Start"],
  "seasonality_expected": true
}}

STEP 7: ENHANCE GUARDRAILS
For each guardrail, add executable flags:
- rule_type: "inverted_rank" | "data_quality" | "business_logic"
- column: Affected column name
- description: Human-readable rule
- validation_logic: Pseudo-code or constraint
- invert_axis: true if lower values are better (for plotting)
- filter_condition: SQL-like filter if needed (e.g., "position <= 10")
- transformation: LOG, SQRT, etc. if needed

Example:
{{
  "rule_type": "inverted_rank",
  "column": "our_position",
  "description": "Position is a rank where lower is better (1 > 10)",
  "validation_logic": "When calculating correlation, invert sign for position",
  "invert_axis": true,
  "filter_condition": "our_position <= 10",
  "transformation": null
}}

STEP 8: OUTPUT JSON
{{
  "reasoning": "your 3-sentence deduction from Step 1",
  "business_model": "E-commerce | SaaS | Marketplace | Advertising | B2B | Retail | Other",
  "industry": "specific industry",
  "business_type": "specific operation type",
  "top_kpis": [
    {{
      "name": "KPI name",
      "kpi_type": "one of the KPI types",
      "formula_template": "one of the formula templates",
      "numerator": "column_name",
      "numerator_agg": "SUM | MEAN | etc.",
      "denominator": "column_name or null",
      "denominator_agg": "SUM | MEAN | etc. or null",
      "supporting_columns": ["col1", "col2"],
      "executable_formula": "SUM(col1) / SUM(col2)"
    }}
  ],
  "confidence": 0.0-1.0,
  "additional_context": "insights about the business domain",
  "recommended_lenses": [
    {{
      "lens_name": "...",
      "priority": 1,
      "objective": "...",
      "supporting_columns": [...],
      "confidence": 0.0-1.0
    }}
  ],
  "eam_mapping": [
    {{
      "anchor": "...",
      "anchor_column": "...",
      "attributes": [...],
      "measures": [
        {{
          "column": "...",
          "aggregation_type": "SUM | MEAN | LAST | etc.",
          "rationale": "..."
        }}
      ],
      "relationships": {{...}},
      "cardinality": {{...}}
    }}
  ],
  "temporal_behavior": {{
    "primary_periodicity": "Daily | Weekly | Monthly | Quarterly | Yearly",
    "rationale": "...",
    "critical_slices": [...],
    "seasonality_expected": true | false
  }},
  "guardrails": [
    {{
      "rule_type": "...",
      "column": "...",
      "description": "...",
      "validation_logic": "...",
      "invert_axis": true | false,
      "filter_condition": "..." or null,
      "transformation": "..." or null
    }}
  ]
}}

=== CRITICAL RULES ===

1. **Generic Column Names**: Use statistical fingerprints to infer meaning:
   - Range (0.05-500): Price, Age, Percentage
   - Mean: Distinguishes Count (mean=10) vs Price (mean=49.99)
   - Unique Ratio: 0.99 = Identifier, 0.05 = Category
   - Distribution "sparse" + 70% zeros: Event Trigger

2. **Lens Budget**: You MUST recommend EXACTLY {lens_budget} lenses (no more, no less)

3. **EAM Validation**: Identifiers and categories are NEVER measures

4. **Guardrails**: Include at least 1 guardrail if you detect:
   - Rank/position columns
   - Potential data quality issues
   - Business logic constraints

5. **Confidence**: Reflect data quality, column clarity, and semantic ambiguity

Return ONLY valid JSON."""
        
        return prompt
    
    def _validate_kpis_by_type(self, kpi_list: List[Dict], profile: SemanticProfile) -> List[str]:
        """
        Validate KPIs using semantic requirement validation (not hardcoded dictionary).
        
        FIX FOR FLAW B: Validates based on KPI type requirements, not KPI names.
        
        Args:
            kpi_list: List of KPI dicts with {kpi, kpi_type, supporting_columns, formula}
            profile: Semantic profile
            
        Returns:
            List of validated KPI names
        """
        # Extract available semantic roles
        available_roles = {}
        for entity in profile.entities:
            role = entity.semantic_guess
            if role not in available_roles:
                available_roles[role] = 0
            available_roles[role] += 1
        
        validated = []
        
        for kpi_data in kpi_list:
            # Handle both dict and string formats
            if isinstance(kpi_data, str):
                # Fallback: treat as generic KPI
                kpi_name = kpi_data
                kpi_type = "financial_basic"
                formula = "SUM"
            else:
                kpi_name = kpi_data.get("kpi", "Unknown")
                kpi_type = kpi_data.get("kpi_type")
                formula = kpi_data.get("formula")
            
            # Validate KPI type
            if kpi_type not in KPI_TYPE_REQUIREMENTS:
                logger.warning(f"✗ KPI '{kpi_name}' rejected (unknown type: {kpi_type})")
                continue
            
            requirements = KPI_TYPE_REQUIREMENTS[kpi_type]
            required_roles = requirements["required_roles"]
            min_count = requirements.get("min_count", 1)
            
            # Check if required roles exist with sufficient count
            matching_count = sum(
                available_roles.get(role, 0) 
                for role in required_roles
            )
            
            if matching_count < min_count:
                logger.warning(f"✗ KPI '{kpi_name}' rejected (missing required roles: {required_roles}, need {min_count}, have {matching_count})")
                continue
            
            # Validate formula template
            if formula and formula not in FORMULA_LIBRARY:
                logger.warning(f"✗ KPI '{kpi_name}' rejected (unsupported formula: {formula})")
                continue
            
            validated.append(kpi_name)
            logger.info(f"✓ KPI '{kpi_name}' validated (type={kpi_type}, formula={formula})")
        
        return validated
    
    def _extract_json_robustly(self, content: str) -> Dict[str, Any]:
        """
        Robust JSON extraction from LLM response.
        
        Handles:
        - Markdown code blocks
        - Leading/trailing text
        - Multiple JSON objects (takes first)
        """
        # Try to extract from markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        # Try direct parsing first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object with regex
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Last resort: try to extract key-value pairs manually
        logger.warning("Failed all JSON parsing methods")
        return None
