"""
Phase 1: Council of Lenses (Refactored for Analytical Engineering)

Transforms from "Creative Writing" to "Analytical Engineering" by:
1. Accepting LensRecommendation from Phase 0.5 (Strategic Alignment)
2. Banning "total" keyword, enforcing aggregation_scope (Denominator Precision)
3. Injecting guardrails into prompts (Boundary Awareness)
4. Implementing deduplication (Analytical Redundancy Fix)
5. Priority-based budgeting (Resource Allocation)
"""
import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from utils.schemas import (
    BusinessContext, SemanticProfile, Hypothesis, 
    LensRecommendation, DomainGuardrail, AggregationScope
)
from utils.helpers import get_logger

load_dotenv()
logger = get_logger(__name__)


class LensAgent:
    """
    Refactored Lens Agent for Analytical Engineering.
    
    Key Changes:
    - Accepts LensRecommendation from Phase 0.5
    - Injects objective and supporting_columns
    - Passes guardrails to LLM
    - Implements priority-based budgeting
    """
    
    def __init__(
        self,
        lens_recommendation: LensRecommendation,
        business_context: BusinessContext,
        profile: SemanticProfile,
        guardrails: List[DomainGuardrail]
    ):
        """
        Initialize lens agent with Phase 0.5 context.
        
        Args:
            lens_recommendation: Specific lens from Phase 0.5
            business_context: Business context with EAM mapping
            profile: Semantic profile
            guardrails: Guardrails from Phase 0.5
        """
        self.lens_rec = lens_recommendation
        self.business_context = business_context
        self.profile = profile
        self.guardrails = guardrails
        
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.client = Groq(api_key=self.api_key)
        
        # Calculate hypothesis budget based on priority
        self.hypothesis_budget = self._calculate_hypothesis_budget()
        
        logger.info(
            f"Initialized {self.lens_rec.lens_name} Agent "
            f"(Priority {self.lens_rec.priority}, Budget: {self.hypothesis_budget} hypotheses)"
        )
    
    def _calculate_hypothesis_budget(self) -> int:
        """
        Calculate number of hypotheses to generate based on priority.
        
        Priority 1 (Critical): 6 hypotheses
        Priority 2: 5 hypotheses
        Priority 3 (Medium): 3 hypotheses
        Priority 4: 2 hypotheses
        Priority 5+ (Low): 1 hypothesis
        """
        budget_map = {
            1: 6,
            2: 5,
            3: 3,
            4: 2
        }
        return budget_map.get(self.lens_rec.priority, 1)
    
    def _format_guardrails(self) -> str:
        """Format guardrails for prompt injection."""
        if not self.guardrails:
            return "None"
        
        lines = []
        for gr in self.guardrails:
            lines.append(f"- [{gr.rule_type}] {gr.column}: {gr.description}")
            if gr.invert_axis:
                lines.append(f"  ⚠️ CRITICAL: Lower values are better for {gr.column}")
            if gr.filter_condition:
                lines.append(f"  📊 Filter: {gr.filter_condition}")
        
        return "\n".join(lines)
    
    def _build_system_prompt(self) -> str:
        """
        Build dynamic system prompt with Phase 0.5 context.
        
        Flaw 1 Fix: Injects specific objective from Phase 0.5
        Flaw 3 Fix: Injects guardrails
        """
        # Get temporal behavior if available
        temporal_info = ""
        if self.business_context.temporal_behavior:
            tb = self.business_context.temporal_behavior
            temporal_info = f"""
TEMPORAL GUIDANCE:
- Primary Periodicity: {tb.primary_periodicity}
- Rationale: {tb.rationale}
- Critical Slices: {', '.join(tb.critical_slices)}
- Seasonality Expected: {tb.seasonality_expected}
"""
        
        return f"""You are the {self.lens_rec.lens_name} Agent.

STRATEGIC OBJECTIVE (from Phase 0.5):
{self.lens_rec.objective}

Every hypothesis you generate MUST serve this specific goal.

AVAILABLE COLUMNS (supporting this lens):
{', '.join(self.lens_rec.supporting_columns)}

CRITICAL: You may ONLY use columns listed above. Do not reference any other columns.

DATASET LAWS & GUARDRAILS:
{self._format_guardrails()}
{temporal_info}

CRITICAL RULES:
1. Do NOT use the word "total" - specify aggregation_scope instead
2. All hypotheses must validate against guardrails
3. Use only the columns listed in AVAILABLE COLUMNS
4. Every hypothesis MUST have aggregation_scope (GLOBAL, TEMPORAL, or DIMENSIONAL)
"""
    
    def _build_user_prompt(self) -> str:
        """
        Build user prompt with analytical engineering requirements.
        
        Flaw 2 Fix: Bans "total", enforces aggregation_scope
        """
        return f"""Generate exactly {self.hypothesis_budget} high-impact hypotheses for the {self.lens_rec.lens_name} lens.

Business Context:
- Industry: {self.business_context.industry}
- Business Type: {self.business_context.business_type}

For each hypothesis, provide:
{{
  "title": "Brief hypothesis title",
  "business_metric": "Column name from available columns (NOT abstract concept)",
  "aggregation_scope": "GLOBAL | TEMPORAL | DIMENSIONAL",
  "time_grain": "daily | weekly | monthly (if TEMPORAL, else null)",
  "dimensions": ["column1", "column2"],
  "metric_template": "SUM | RATIO | GROWTH | MEAN | etc.",
  "numerator_concept": "column_name",
  "denominator_concept": "column_name or null",
  "guardrail_applied": "inverted_rank | data_quality | business_logic (if applicable, else null)",
  "priority": 1-10 (1 = highest),
  "confidence": 0.0-1.0,
  "description": "What we're testing and why"
}}

DENOMINATOR RULES (CRITICAL):
- NEVER use "total" as a denominator
- If you want a ratio, specify:
  - aggregation_scope: GLOBAL (entire dataset) | TEMPORAL (time-based) | DIMENSIONAL (category-based)
  - If DIMENSIONAL, specify target dimension in dimensions array

WRONG: denominator_concept="total sales"
RIGHT: numerator_concept="sales", denominator_concept=null, aggregation_scope="GLOBAL"
RIGHT: numerator_concept="sales", denominator_concept="adcost", aggregation_scope="DIMENSIONAL", dimensions=["category"]

GUARDRAIL VALIDATION:
- If a hypothesis involves a column with guardrails, set guardrail_applied
- Example: If using "our_position" and there's an inverted_rank guardrail, set guardrail_applied="inverted_rank"

Return JSON array of exactly {self.hypothesis_budget} hypotheses, ordered by priority (highest first)."""
    
    def generate_hypotheses(self) -> List[Hypothesis]:
        """Generate hypotheses using Phase 0.5 context."""
        logger.info(f"{self.lens_rec.lens_name} generating {self.hypothesis_budget} hypotheses")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": self._build_user_prompt()}
                ],
                temperature=0.4,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            hypotheses_data = self._extract_json_robustly(content)
            
            if not hypotheses_data:
                logger.error(f"{self.lens_rec.lens_name}: Failed to extract JSON")
                return []
            
            hypotheses = []
            for h in hypotheses_data[:self.hypothesis_budget]:
                try:
                    # Validate required fields
                    if not all(k in h for k in ['title', 'business_metric', 'aggregation_scope', 'numerator_concept']):
                        logger.warning(f"Skipping hypothesis with missing required fields: {h.get('title', 'Unknown')}")
                        continue
                    
                    # Check for "total" hallucination
                    if h.get('denominator_concept') and 'total' in h['denominator_concept'].lower():
                        logger.warning(f"Rejected hypothesis with 'total' in denominator: {h['title']}")
                        continue
                    
                    # Generate ID
                    timestamp = datetime.now().strftime("%Y%m%d")
                    metric_slug = h['business_metric'].replace(' ', '_').lower()
                    hyp_id = f"{self.lens_rec.lens_name.replace(' ', '_').lower()}_{timestamp}_{metric_slug}"
                    
                    hypothesis = Hypothesis(
                        id=hyp_id,
                        lens=self.lens_rec.lens_name,
                        title=h['title'],
                        business_metric=h['business_metric'],
                        aggregation_scope=AggregationScope(h['aggregation_scope']),
                        time_grain=h.get('time_grain'),
                        dimensions=h.get('dimensions', []),
                        metric_template=h.get('metric_template', 'SUM'),
                        numerator_concept=h['numerator_concept'],
                        denominator_concept=h.get('denominator_concept'),
                        guardrail_applied=h.get('guardrail_applied'),
                        priority=h.get('priority', self.lens_rec.priority),
                        confidence=h.get('confidence', self.lens_rec.confidence),
                        description=h.get('description', ''),
                        required_semantic_roles=[],  # Legacy
                        expected_insight_type=h.get('expected_insight_type', 'trend')
                    )
                    hypotheses.append(hypothesis)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse hypothesis: {e}")
                    continue
            
            logger.info(f"{self.lens_rec.lens_name} generated {len(hypotheses)} validated hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"{self.lens_rec.lens_name} failed: {e}")
            return []
    
    def _extract_json_robustly(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Robust JSON extraction from LLM response."""
        # Try to extract from markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        content = content.strip()
        
        # Try direct parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try regex
        json_pattern = r'\[[^\[\]]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^\[\]]*)*\]'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        logger.warning("Failed all JSON parsing methods")
        return None


class CouncilOfLenses:
    """
    Orchestrates lenses with Phase 0.5 integration and deduplication.
    
    Key Changes:
    - Creates one LensAgent per Phase 0.5 recommendation
    - Implements deduplication (Flaw 4 fix)
    - Sorts by priority
    """
    
    def __init__(self):
        pass
    
    def generate_all_hypotheses(
        self,
        business_context: BusinessContext,
        profile: SemanticProfile
    ) -> List[Hypothesis]:
        """
        Generate hypotheses from all recommended lenses.
        
        Flaw 1 Fix: Uses recommended_lenses from Phase 0.5
        Flaw 4 Fix: Deduplicates hypotheses
        Flaw 5 Fix: Priority-based budgeting
        """
        logger.info("Council of Lenses generating hypotheses from Phase 0.5 recommendations")
        
        if not business_context.recommended_lenses:
            logger.warning("No recommended lenses from Phase 0.5, cannot generate hypotheses")
            return []
        
        all_hypotheses = []
        
        # Create one LensAgent per recommendation
        for lens_rec in business_context.recommended_lenses:
            agent = LensAgent(
                lens_recommendation=lens_rec,
                business_context=business_context,
                profile=profile,
                guardrails=business_context.guardrails
            )
            hypotheses = agent.generate_hypotheses()
            all_hypotheses.extend(hypotheses)
        
        # Deduplicate (Flaw 4 fix)
        deduplicated = self._deduplicate_hypotheses(all_hypotheses)
        
        # Sort by priority (highest first)
        deduplicated.sort(key=lambda x: (x.priority, x.confidence), reverse=False)  # Lower priority number = higher priority
        
        logger.info(
            f"Council generated {len(all_hypotheses)} total hypotheses, "
            f"{len(deduplicated)} after deduplication "
            f"(removed {len(all_hypotheses) - len(deduplicated)} duplicates)"
        )
        
        return deduplicated
    
    def _deduplicate_hypotheses(self, all_hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Remove duplicate hypotheses using semantic similarity.
        
        Two hypotheses are duplicates if:
        1. Same business_metric
        2. Same dimensions (order-independent)
        3. Same aggregation_scope
        4. Same time_grain (for TEMPORAL scope) - Shadow Flaw Fix
        
        This ensures "Daily Sales Trend" and "Weekly Sales Trend" are treated as unique insights.
        """
        seen = {}
        deduplicated = []
        
        for hyp in all_hypotheses:
            # Create signature (Shadow Flaw Fix: include time_grain)
            dims_sorted = tuple(sorted(hyp.dimensions))
            signature = (
                hyp.business_metric,
                dims_sorted,
                hyp.aggregation_scope,
                hyp.time_grain  # NEW: Prevents merging daily vs weekly trends
            )
            
            if signature in seen:
                # Merge: keep higher priority (lower number)
                existing = seen[signature]
                if hyp.priority < existing.priority:
                    logger.info(f"Replacing duplicate: '{existing.title}' with '{hyp.title}' (higher priority)")
                    seen[signature] = hyp
                else:
                    logger.info(f"Skipping duplicate: '{hyp.title}' (lower priority than '{existing.title}')")
            else:
                seen[signature] = hyp
        
        return list(seen.values())


# Entry point function
def generate_hypotheses(
    business_context: BusinessContext,
    profile: SemanticProfile
) -> List[Hypothesis]:
    """
    Generate hypotheses using Phase 0.5 analytical blueprint.
    
    This is the main entry point for Phase 1.
    """
    council = CouncilOfLenses()
    return council.generate_all_hypotheses(business_context, profile)
