"""
Phase 1: Council of Lenses (Level 10 Intelligence)

Enhancements:
1. Flaw G Fix: Denominator scope (SAME_ROW, GLOBAL_SUM, GROUP_SUM)
2. Flaw H Fix: Cross-metric hypotheses with 2+ dimensions for P1
3. Flaw I Fix: Explicit guardrail transformations
4. Rich schema injection with dtype, semantic role, min/max/unique
5. Diversity sampling (10 rows: start, middle, end)
6. Statistical anchors for context
"""
import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
from utils.schemas import (
    BusinessContext, SemanticProfile, Hypothesis, 
    LensRecommendation, DomainGuardrail, AggregationScope
)
from utils.helpers import get_logger

load_dotenv()
logger = get_logger(__name__)


class LensAgent:
    """
    Level 10 Intelligence Lens Agent.
    
    Key Enhancements:
    - Diversity sampling (10 rows)
    - Rich schema injection (dtype, semantic role, stats)
    - Explicit ratio syntax (denominator scope)
    - Friction-first prompting
    - Multi-dimensional enforcement for P1
    """
    
    def __init__(
        self,
        lens_recommendation: LensRecommendation,
        business_context: BusinessContext,
        profile: SemanticProfile,
        guardrails: List[DomainGuardrail],
        df: pd.DataFrame  # NEW: Need dataframe for sampling
    ):
        """
        Initialize lens agent with Phase 0.5 context and data.
        
        Args:
            lens_recommendation: Specific lens from Phase 0.5
            business_context: Business context with EAM mapping
            profile: Semantic profile
            guardrails: Guardrails from Phase 0.5
            df: Original dataframe for sampling
        """
        self.lens_rec = lens_recommendation
        self.business_context = business_context
        self.profile = profile
        self.guardrails = guardrails
        self.df = df
        
        # Use GROQ_API_KEY_2 for Phase 1 (rate limit distribution)
        self.api_key = os.getenv("GROQ_API_KEY_2")
        if not self.api_key:
            logger.warning("GROQ_API_KEY_2 not found, falling back to GROQ_API_KEY_3")
            self.api_key = os.getenv("GROQ_API_KEY_3")
            if not self.api_key:
                logger.error("GROQ_API_KEY_3 also not found! Please set at least one API key.")
                raise ValueError("No valid Groq API key found for Phase 1")
        
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.client = Groq(api_key=self.api_key)
        
        # Calculate hypothesis budget based on priority
        self.hypothesis_budget = self._calculate_hypothesis_budget()
        
        # Safety fallback state
        self.correction_notes = None  # Will be set if retry is needed
        self.retry_count = 0
        self.max_retries = 1  # Only retry once
        
        logger.info(
            f"Initialized {self.lens_rec.lens_name} Agent "
            f"(Priority {self.lens_rec.priority}, Budget: {self.hypothesis_budget} hypotheses)"
        )
    
    def _calculate_hypothesis_budget(self) -> int:
        """Calculate number of hypotheses based on priority (INCREASED BUDGET)."""
        budget_map = {1: 10, 2: 8, 3: 5, 4: 3}
        return budget_map.get(self.lens_rec.priority, 2)
    
    def _generate_diverse_sample(self) -> List[Dict[str, Any]]:
        """
        Generate diverse sample of 10 rows (start, middle, end).
        
        Returns:
            List of row dictionaries
        """
        # Convert to pandas if needed
        if hasattr(self.df, 'to_pandas'):
            # Polars DataFrame
            df_pd = self.df.to_pandas()
        else:
            # Already pandas
            df_pd = self.df
        
        n_rows = len(df_pd)
        if n_rows <= 10:
            sample_df = df_pd
        else:
            # 3 from start, 4 from middle, 3 from end
            start_idx = list(range(0, min(3, n_rows)))
            middle_idx = list(range(n_rows // 2 - 2, n_rows // 2 + 2))
            end_idx = list(range(max(0, n_rows - 3), n_rows))
            
            all_idx = start_idx + middle_idx + end_idx
            sample_df = df_pd.iloc[all_idx]
        
        # Convert to dict, limit to supporting columns
        sample_data = []
        for _, row in sample_df.head(10).iterrows():
            row_dict = {}
            for col in self.lens_rec.supporting_columns:
                if col in row:
                    val = row[col]
                    # Convert to JSON-serializable
                    if pd.isna(val):
                        row_dict[col] = None
                    elif isinstance(val, (pd.Timestamp, datetime)):
                        row_dict[col] = str(val)
                    else:
                        row_dict[col] = val
            sample_data.append(row_dict)
        
        return sample_data
    
    def _build_column_metadata(self) -> str:
        """
        Build rich column metadata with dtype, semantic role, and stats.
        
        Returns:
            Formatted string with column metadata
        """
        # Convert to pandas if needed
        if hasattr(self.df, 'to_pandas'):
            df_pd = self.df.to_pandas()
        else:
            df_pd = self.df
        
        lines = []
        
        # UPGRADE 1: LITERAL NAME LAW - Make column names explicit
        for col in self.lens_rec.supporting_columns:
            if col not in df_pd.columns:
                continue
            
            # Get dtype
            dtype = str(df_pd[col].dtype)
            
            # Get semantic role from profile
            semantic_role = "unknown"
            for entity in self.profile.entities:
                if hasattr(entity, 'column_name') and entity.column_name == col:
                    if hasattr(entity, 'semantic_guess'):
                        semantic_role = entity.semantic_guess
                    break
            
            # Get stats
            stats_str = ""
            if pd.api.types.is_numeric_dtype(df_pd[col]):
                min_val = df_pd[col].min()
                max_val = df_pd[col].max()
                mean_val = df_pd[col].mean()
                stats_str = f" [min: {min_val:.2f}, max: {max_val:.2f}, mean: {mean_val:.2f}]"
            else:
                unique_count = df_pd[col].nunique()
                stats_str = f" [unique: {unique_count}]"
            
            # Check for guardrails
            guardrail_note = ""
            for gr in self.guardrails:
                if gr.column == col:
                    if gr.invert_axis:
                        guardrail_note = " ⚠️ INVERTED RANK"
                    break
            
            # UPGRADE 1: Use EXACT_KEY format to enforce literal names
            lines.append(f"- EXACT_KEY: \"{col}\" (Type: {dtype}, Role: {semantic_role}){stats_str}{guardrail_note}")
        
        return "\n".join(lines)
    
    def _format_guardrails(self) -> str:
        """Format guardrails with explicit transformations."""
        if not self.guardrails:
            return "None"
        
        # Convert to pandas if needed
        if hasattr(self.df, 'to_pandas'):
            df_pd = self.df.to_pandas()
        else:
            df_pd = self.df
        
        lines = []
        for gr in self.guardrails:
            lines.append(f"- [{gr.rule_type}] {gr.column}: {gr.description}")
            if gr.invert_axis:
                # Calculate max for transformation
                if gr.column in df_pd.columns:
                    max_val = int(df_pd[gr.column].max())
                    lines.append(f"  🔧 Transformation: INVERT({gr.column}) = ({max_val + 1} - {gr.column})")
                else:
                    lines.append(f"  🔧 Transformation: INVERT({gr.column}) = (MAX + 1 - {gr.column})")
            if gr.filter_condition:
                lines.append(f"  📊 Filter: {gr.filter_condition}")
        
        return "\n".join(lines)
    
    def _build_system_prompt(self) -> str:
        """
        Build Level 10 system prompt with rich context.
        """
        # UPGRADE 2: GHOST COLUMN BAN - Build temporal warning
        temporal_warning = ""
        if self.business_context.temporal_behavior:
            if hasattr(self.business_context.temporal_behavior, 'critical_slices'):
                temporal_warning = "\n\nCRITICAL SLICES (Future Intents - DO NOT USE as dimensions):\n"
                for slice_name in self.business_context.temporal_behavior.critical_slices:
                    temporal_warning += f"- {slice_name} (will be engineered in Phase 4, not available now)\n"
        
        # SAFETY FALLBACK: Add correction notes if this is a retry
        correction_section = ""
        if self.correction_notes:
            correction_section = f"""\n\n=== CORRECTION FROM PREVIOUS ATTEMPT ===
[CRITICAL] Your previous hypotheses had errors. Here's what went wrong:
{self.correction_notes}

Please generate NEW hypotheses that avoid these mistakes!
==================================\n\n"""
        
        column_metadata = self._build_column_metadata()
        guardrail_str = self._format_guardrails()
        
        return f"""You are the {self.lens_rec.lens_name} Agent - a Level 10 Analytical Intelligence system.
{correction_section}
STRATEGIC OBJECTIVE (from Phase 0.5):
{self.lens_rec.objective}

Every hypothesis you generate MUST serve this specific goal and find "hidden money," not just repeat column headers.

AVAILABLE COLUMNS (supporting this lens):
{column_metadata}
{temporal_warning}
DATASET LAWS & GUARDRAILS:
{guardrail_str}

TEMPORAL GUIDANCE:
- Primary Periodicity: {self.business_context.temporal_behavior.primary_periodicity if self.business_context.temporal_behavior else 'Unknown'}
- Seasonality Expected: {self.business_context.temporal_behavior.seasonality_expected if self.business_context.temporal_behavior else 'Unknown'}

=== THE LITERAL NAME LAW ===
[CRITICAL] You are STRICTLY FORBIDDEN from changing column name casing!
- If the column is "Category", you MUST write "Category" (not "category")
- If the column is "Our_Position", you MUST write "Our_Position" (not "our_position")
- Use ONLY the EXACT_KEY names from AVAILABLE COLUMNS above
- Do NOT use Critical Slices as dimensions - they don't exist yet!

CRITICAL RULES FOR LEVEL 10 INTELLIGENCE:

1. DENOMINATOR PRECISION (Flaw G Fix):
   ❌ WRONG: numerator="sales", denominator="sales", scope=SAME_ROW (equals 1!)
   ✅ RIGHT: numerator="sales", denominator="sales", scope=GLOBAL_SUM (market share)

2. CROSS-METRIC HYPOTHESES (Flaw H Fix):
   ❌ WRONG: "Average sales by category" (boring reporting)
   ✅ RIGHT: "Is ROAS higher for Formal Shoes at Position 1 vs Running Shoes at Position 3?"

3. MULTI-DIMENSIONAL ENFORCEMENT:
   Priority 1 hypotheses MUST involve at least 2 dimensions

4. EXPLICIT GUARDRAIL TRANSFORMATIONS:
   Set guardrail_transformation: "INVERT(our_position) = (11 - our_position)"

5. CROSS-METRIC PAIRING (NEW):
   Try combining every Success Metric (Sales, Clicks, Conversions) with at least TWO different Cost Metrics (AdCost, Price, Cost) to uncover different friction points.
   Example: Sales/AdCost (ROAS), Sales/Price (Price Efficiency), Sales/Cost (Margin Efficiency)

6. DIMENSION PERMUTATION (For Priority 1 Lenses):
   Generate one hypothesis for EACH unique categorical dimension available.
   Example: If you have Category, SubCategory, Brand → create separate hypotheses for each
OUTPUT FORMAT:
Return ONLY a JSON array of hypothesis objects. Each must have:
- title: Brief diagnostic title
- business_metric: Column name from AVAILABLE COLUMNS
- aggregation_scope: GLOBAL | TEMPORAL | DIMENSIONAL
- time_grain: "daily" | "weekly" | "monthly" (if TEMPORAL)
- dimensions: List of EXACT column names from AVAILABLE COLUMNS
- metric_template: RATIO | MEAN | SUM | GROWTH
- numerator_concept: Business concept for numerator
- denominator_concept: Business concept for denominator (if RATIO)
- denominator_scope: SAME_ROW | GLOBAL_SUM | GROUP_SUM (if RATIO)
- guardrail_applied: Column name if guardrail applies
- guardrail_transformation: Exact formula if guardrail applies
- description: Brief explanation
- priority: 1-5 (1=critical, 5=exploratory)
- confidence: 0.0-1.0
"""
    
    def _build_user_prompt(self, sample_data: List[Dict[str, Any]]) -> str:
        """
        Build Level 10 user prompt with data preview and friction-first mission.
        """
        return f"""=== DATA PREVIEW (Representative Samples) ===
{json.dumps(sample_data, indent=2)}

=== DATASET CHARACTERISTICS ===
- Total Rows: {len(self.df):,}
- Period: {self.business_context.temporal_behavior.primary_periodicity if self.business_context.temporal_behavior else 'Unknown'}
- Key Grain: {', '.join(self.profile.identifier_columns)}

=== ANALYST MISSION: FIND THE FRICTION ===

Using the sample rows above as your 'Truth', generate exactly {self.hypothesis_budget} hypotheses that find "hidden money."

Example thought process:
"I see Subcategory X has high AdCost in the samples but Sales are low—let's test that correlation globally."

For each hypothesis, provide:
{{
  "title": "Brief hypothesis title (diagnostic, not descriptive)",
  "business_metric": "Column name from available columns",
  "aggregation_scope": "GLOBAL | TEMPORAL | DIMENSIONAL",
  "time_grain": "daily | weekly | monthly (if TEMPORAL, else null)",
  "dimensions": ["column1", "column2"],  // Priority 1 MUST have 2+ dimensions
  "metric_template": "RATIO | GROWTH | MEAN | etc.",
  "numerator_concept": "column_name",
  "denominator_concept": "column_name or null",
  "denominator_scope": "SAME_ROW | GLOBAL_SUM | GROUP_SUM (if denominator exists)",
  "guardrail_applied": "inverted_rank | data_quality | business_logic (if applicable)",
  "guardrail_transformation": "Exact formula (e.g., 'INVERT(our_position) = (11 - our_position)')",
  "priority": {self.lens_rec.priority},
  "confidence": 0.0-1.0,
  "description": "What friction point this tests and why"
}}

CRITICAL VALIDATION:
- If numerator == denominator, denominator_scope MUST be GLOBAL_SUM or GROUP_SUM (never SAME_ROW)
- Priority {self.lens_rec.priority} hypotheses with priority 1 MUST have len(dimensions) >= 2
- All guardrails MUST have explicit guardrail_transformation formulas

Return JSON array of exactly {self.hypothesis_budget} hypotheses, ordered by priority (highest first)."""
    
    
    def retry_with_corrections(self, correction_notes: str) -> List[Hypothesis]:
        """
        Retry hypothesis generation with correction notes about previous errors.
        
        Args:
            correction_notes: Description of what went wrong (e.g., hallucinated columns)
            
        Returns:
            List of corrected hypotheses
        """
        if self.retry_count >= self.max_retries:
            logger.warning(f"{self.lens_rec.lens_name}: Max retries reached, cannot retry again")
            return []
        
        logger.info(f"{self.lens_rec.lens_name}: Retrying with corrections...")
        self.correction_notes = correction_notes
        self.retry_count += 1
        
        # Generate new hypotheses with correction notes injected
        return self.generate_hypotheses()
    
    def generate_hypotheses(self) -> List[Hypothesis]:
        """Generate Level 10 hypotheses using rich context."""
        logger.info(f"{self.lens_rec.lens_name} generating {self.hypothesis_budget} Level 10 hypotheses")
        
        try:
            # Generate diverse sample
            sample_data = self._generate_diverse_sample()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": self._build_user_prompt(sample_data)}
                ],
                temperature=0.4,
                max_tokens=3000  # Increased for richer output
            )
            
            content = response.choices[0].message.content
            hypotheses_data = self._extract_json_robustly(content)
            
            if not hypotheses_data:
                logger.error(f"{self.lens_rec.lens_name}: Failed to extract JSON")
                return []
            
            hypotheses = []
            
            # UPGRADE 3: CASE-SENSITIVITY REPAIRMAN - Build column name map
            valid_cols = {c.lower(): c for c in self.df.columns}
            
            for h in hypotheses_data[:self.hypothesis_budget]:
                try:
                    # Validate required fields
                    required = ['title', 'business_metric', 'aggregation_scope', 'numerator_concept']
                    if not all(k in h for k in required):
                        logger.warning(f"Skipping hypothesis with missing fields: {h.get('title', 'Unknown')}")
                        continue
                    
                    # UPGRADE 3: Auto-repair dimension casing
                    fixed_dimensions = []
                    for dim in h.get('dimensions', []):
                        if dim in self.df.columns:
                            # Already perfect
                            fixed_dimensions.append(dim)
                        elif dim.lower() in valid_cols:
                            # Auto-repair casing: 'category' -> 'Category'
                            correct_name = valid_cols[dim.lower()]
                            logger.info(f"Auto-repaired dimension: '{dim}' → '{correct_name}'")
                            fixed_dimensions.append(correct_name)
                        else:
                            # It's a ghost column (like Is_Weekend), drop it
                            logger.warning(f"Dropping ghost dimension '{dim}' from hypothesis: {h['title']}")
                    
                    h['dimensions'] = fixed_dimensions
                    
                    # Flaw G validation: Check for numerator==denominator trap
                    if h.get('denominator_concept') == h.get('numerator_concept'):
                        if h.get('denominator_scope') == 'SAME_ROW':
                            logger.warning(f"Rejected: {h['title']} has numerator==denominator with SAME_ROW scope")
                            continue
                        if not h.get('denominator_scope'):
                            logger.warning(f"Rejected: {h['title']} has numerator==denominator but no scope specified")
                            continue
                    
                    # Flaw H validation: P1 should have 2+ dimensions (warning only, not rejection)
                    if self.lens_rec.priority == 1:
                        if len(h.get('dimensions', [])) < 2:
                            logger.warning(f"P1 hypothesis has <2 dimensions (acceptable): {h['title']}")
                    
                    # Check for "total" hallucination
                    if h.get('denominator_concept') and 'total' in h['denominator_concept'].lower():
                        logger.warning(f"Rejected hypothesis with 'total' in denominator: {h['title']}")
                        continue
                    
                    # Generate unique ID with title slug
                    timestamp = datetime.now().strftime("%Y%m%d")
                    metric_slug = h['business_metric'].replace(' ', '_').lower()
                    
                    # Create title slug (first 3 meaningful words)
                    title_words = h['title'].lower().replace('-', ' ').split()
                    # Filter out common words
                    stop_words = {'by', 'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
                    meaningful_words = [w for w in title_words if w not in stop_words][:3]
                    title_slug = '_'.join(meaningful_words) if meaningful_words else 'hypothesis'
                    
                    hyp_id = f"{self.lens_rec.lens_name.replace(' ', '_').lower()}_{timestamp}_{metric_slug}_{title_slug}"
                    
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
                        denominator_scope=h.get('denominator_scope'),  # NEW
                        guardrail_applied=h.get('guardrail_applied'),
                        guardrail_transformation=h.get('guardrail_transformation'),  # NEW
                        priority=h.get('priority', self.lens_rec.priority),
                        confidence=h.get('confidence', self.lens_rec.confidence),
                        description=h.get('description', ''),
                        required_semantic_roles=[],
                        expected_insight_type=h.get('expected_insight_type', 'efficiency')
                    )
                    hypotheses.append(hypothesis)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse hypothesis: {e}")
                    continue
            
            logger.info(f"{self.lens_rec.lens_name} generated {len(hypotheses)} validated Level 10 hypotheses")
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
    Orchestrates lenses with Level 10 intelligence.
    """
    
    def __init__(self):
        pass
    
    def generate_all_hypotheses(
        self,
        business_context: BusinessContext,
        profile: SemanticProfile,
        df: pd.DataFrame  # NEW: Need dataframe for sampling
    ) -> List[Hypothesis]:
        """
        Generate Level 10 hypotheses from all recommended lenses.
        """
        logger.info("Council of Lenses generating Level 10 hypotheses")
        
        if not business_context.recommended_lenses:
            logger.warning("No recommended lenses from Phase 0.5")
            return []
        
        all_hypotheses = []
        
        # Create one LensAgent per recommendation
        for lens_rec in business_context.recommended_lenses:
            agent = LensAgent(
                lens_recommendation=lens_rec,
                business_context=business_context,
                profile=profile,
                guardrails=business_context.guardrails,
                df=df  # Pass dataframe
            )
            hypotheses = agent.generate_hypotheses()
            all_hypotheses.extend(hypotheses)
        
        # Deduplicate
        deduplicated = self._deduplicate_hypotheses(all_hypotheses)
        
        # Sort by priority
        deduplicated.sort(key=lambda x: (x.priority, x.confidence), reverse=False)
        
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
        4. Same time_grain (Shadow Flaw Fix)
        """
        seen = {}
        
        for hyp in all_hypotheses:
            # Create signature (Shadow Flaw Fix: include time_grain)
            dims_sorted = tuple(sorted(hyp.dimensions))
            signature = (
                hyp.business_metric,
                dims_sorted,
                hyp.aggregation_scope,
                hyp.time_grain  # Prevents merging daily vs weekly trends
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
    profile: SemanticProfile,
    df: pd.DataFrame  # NEW: Need dataframe for sampling
) -> List[Hypothesis]:
    """
    Generate Level 10 hypotheses using Phase 0.5 analytical blueprint.
    
    This is the main entry point for Phase 1.
    """
    council = CouncilOfLenses()
    return council.generate_all_hypotheses(business_context, profile, df)
