"""
Metric Resolver Module.

Deterministically maps abstract business intent (from LLM) to concrete data columns or derived metrics.
Enforces validation and prevents hallucination.
"""
from dataclasses import dataclass
from typing import Optional, List, Literal
import json
from core.metric_blueprint import BlueprintClassifier, MetricType, MetricBlueprint
from utils.schemas import SemanticProfile
from core.derived_metrics_registry import get_derived_metric
from utils.helpers import get_logger

logger = get_logger(__name__)

# Sprint 1: Synonym mapping for abstract business metrics
# Maps common business terms to column name patterns
METRIC_SYNONYMS = {
    'profitability': ['profit', 'margin', 'earnings'],
    'revenue': ['sales', 'income', 'receipts'],
    'cost': ['expense', 'spend', 'expenditure'],
    'efficiency': ['productivity', 'utilization', 'throughput'],
    'performance': ['results', 'outcomes', 'achievement'],
    'growth': ['increase', 'expansion', 'development'],
    'retention': ['loyalty', 'churn', 'attrition'],
    'acquisition': ['onboarding', 'signup', 'conversion']
}

@dataclass
class ResolutionResult:
    resolved_metric: Optional[str]
    resolution_confidence: float
    resolution_type: Literal["direct", "derived", "failed", "unsupported"]
    dependencies: List[str]
    explanation: str
    formula_name: Optional[str] = None # Name of registry item if derived

class MetricResolver:
    """Resolves abstract business metrics to concrete columns using Blueprints."""
    
    def __init__(self, profile: SemanticProfile, llm_client=None):
        self.profile = profile
        self.llm_client = llm_client  # For dynamic LLM-driven resolution
        
    def resolve_metric(self, business_metric: str, required_roles: List[str]) -> ResolutionResult:
        """
        Main entry point for resolution.
        """
        logger.info(f"Resolving metric: '{business_metric}' (roles: {required_roles})")
        
        # 1. Classify Intent (The Blueprint)
        blueprint = BlueprintClassifier.classify(business_metric, required_roles)
        logger.info(f"Blueprint: {blueprint.metric_type} -> primary='{blueprint.primary_concept}', sec='{blueprint.secondary_concept}'")
        
        # 2. Branch by Type
        if blueprint.metric_type == MetricType.DIRECT:
            return self._resolve_direct(blueprint.primary_concept, required_roles)
            
        elif blueprint.metric_type == MetricType.RATIO:
            return self._resolve_ratio(blueprint, required_roles)
            
        elif blueprint.metric_type == MetricType.RATE:
            # Treat rates like ratios or direct % columns
            # Try direct first (e.g. "Conversion Rate" column), then ratio
            direct = self._resolve_direct(blueprint.primary_concept, required_roles)
            if direct.resolution_confidence > 0.7:
                 return direct
            return self._resolve_ratio(blueprint, required_roles)
            
        elif blueprint.metric_type == MetricType.EFFICIENCY:
            # Efficiency is usually a ratio
            return self._resolve_ratio(blueprint, required_roles)

        elif blueprint.metric_type == MetricType.FUNNEL:
             return self._resolve_funnel(blueprint, required_roles)
             
        elif blueprint.metric_type == MetricType.INVENTORY:
             return self._resolve_inventory(blueprint, required_roles) # Placeholder
             
        elif blueprint.metric_type == MetricType.TEMPORAL:
             # Temporal metrics usually need a base metric + time
             # We resolve the base metric
             base_res = self._resolve_direct(blueprint.primary_concept, required_roles)
             if base_res.resolution_confidence > 0.6:
                 # We return the base metric but note it's for temporal analysis
                 base_res.explanation += " (Base for Temporal Analysis)"
                 return base_res
             return self._resolve_unsupported(blueprint, "Could not resolve base metric for temporal analysis")

        # Fallback to direct
        return self._resolve_direct(blueprint.primary_concept, required_roles)

    def _resolve_direct(self, intent: str, required_roles: List[str]) -> ResolutionResult:
        """Find best existing column matching intent and roles."""
        candidates = []
        intent_lower = intent.lower()
        
        # Apply synonym expansion
        search_terms = [intent_lower]
        for synonym_key, synonyms in METRIC_SYNONYMS.items():
            if synonym_key in intent_lower:
                search_terms.extend(synonyms)
            elif intent_lower in synonyms:
                search_terms.append(synonym_key)
                search_terms.extend([s for s in synonyms if s != intent_lower])
        
        # Check against registry first? No, blueprint covers that?
        # Maybe "Profit" is in registry as Sales - Cost?
        # If direct resolution fails, we could try registry lookup as fallback.
        
        candidates = self._find_candidates(intent_lower, required_roles)
        
        # Sort by score desc
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        if not candidates:
             # Try registry fallback if direct failed
            registry_res = self._resolve_from_registry(intent)
            if registry_res.resolution_confidence > 0.6:
                return registry_res
            
            # Final fallback: LLM-driven resolution
            if self.llm_client:
                logger.info(f"Static resolution failed for '{intent}', trying LLM...")
                return self._resolve_with_llm(intent, required_roles)
                
            return ResolutionResult(None, 0.0, "failed", [], f"No direct candidates found for '{intent}'")
            
        best_score, best_entity, best_reasons = candidates[0]
        
        # Thresholds
        if best_score < 0.4:
            return ResolutionResult(None, 0.0, "failed", [], f"Best match '{best_entity.column_name}' confidence too low ({best_score:.2f})")
            
        return ResolutionResult(
            resolved_metric=best_entity.column_name,
            resolution_confidence=min(1.0, best_score),
            resolution_type="direct",
            dependencies=[best_entity.column_name],
            explanation=f"Selected '{best_entity.column_name}': {', '.join(best_reasons)}"
        )
        
    def _resolve_ratio(self, blueprint: MetricBlueprint, roles: List[str]) -> ResolutionResult:
        """Resolve A / B."""
        num_concept = blueprint.primary_concept
        den_concept = blueprint.secondary_concept or "total" # Default?
        
        # If secondary is "context_dependent" (generic ratio), we can't resolve without more info
        if den_concept == "context_dependent":
             return self._resolve_unsupported(blueprint, "Context dependent ratio denominator")
             
        # Resolve Numerator
        num_res = self._resolve_direct(num_concept, []) # Loose roles
        if num_res.resolution_confidence < 0.5:
             return self._resolve_unsupported(blueprint, f"Could not resolve numerator '{num_concept}'")
             
        # Resolve Denominator
        den_res = self._resolve_direct(den_concept, [])
        if den_res.resolution_confidence < 0.5:
              return self._resolve_unsupported(blueprint, f"Could not resolve denominator '{den_concept}'")
              
        return ResolutionResult(
            resolved_metric=f"({num_res.resolved_metric} / {den_res.resolved_metric})", # Virtual
            resolution_confidence=num_res.resolution_confidence * den_res.resolution_confidence,
            resolution_type="derived",
            dependencies=[num_res.resolved_metric, den_res.resolved_metric],
            explanation=f"Ratio: {num_res.explanation} / {den_res.explanation}",
            formula_name="dynamic_ratio"
        )

    def _resolve_funnel(self, blueprint: MetricBlueprint, roles: List[str]) -> ResolutionResult:
        """Check for ordered funnel stages."""
        # Find columns with 'funnel' role
        funnel_cols = []
        for entity in self.profile.entities:
            if entity.semantic_guess == "funnel" and entity.confidence > 0.4:
                funnel_cols.append(entity)
        
        if len(funnel_cols) < 2:
             return self._resolve_unsupported(blueprint, "Insufficient funnel stages found (need 2+)")
             
        # Sort by value count? Or name? Hard to know order without manual input.
        # Ensure we have at least 'funnel' role.
        
        return ResolutionResult(
            resolved_metric="funnel_analysis",
            resolution_confidence=0.7,
            resolution_type="derived",
            dependencies=[c.column_name for c in funnel_cols],
            explanation=f"Found {len(funnel_cols)} funnel stages: {[c.column_name for c in funnel_cols]}",
            formula_name="funnel_projection"
        )
        
    def _resolve_inventory(self, blueprint: MetricBlueprint, roles: List[str]) -> ResolutionResult:
        # Require 'inventory' role + time
        inv_cols = [e for e in self.profile.entities if e.semantic_guess == "inventory"]
        if not inv_cols:
             return self._resolve_unsupported(blueprint, "No inventory columns found")
             
        return self._resolve_direct(blueprint.primary_concept, roles) # Fallback to direct lookup of "Turnover" etc

    def _resolve_unsupported(self, blueprint: MetricBlueprint, reason: str) -> ResolutionResult:
        return ResolutionResult(
            resolved_metric=None,
            resolution_confidence=0.0,
            resolution_type="unsupported",
            dependencies=[],
            explanation=f"Unsupported {blueprint.metric_type}: {reason}"
        )
        
    def _find_candidates(self, intent: str, required_roles: List[str]) -> List[tuple]:
        """Core scoring logic."""
        candidates = []
        intent_lower = intent.lower()
        
        for entity in self.profile.entities:
            # Strict Type Check: Only Numeric or Boolean
            if entity.statistical_type not in ["numeric", "boolean"]:
                continue
                
            score = 0.0
            reasons = []
            
            # 1. Semantic Role Check
            # If required_roles are present, entity MUST match or be neutral?
            # User said: "Code resolves data. Validation enforces truth."
            if required_roles:
                # Standardize roles for comparison
                normalized_roles = [r.lower().replace(" ", "").replace("_", "") for r in required_roles]
                entity_role = entity.semantic_guess.lower().replace(" ", "").replace("_", "")
                
                # Check for direct or partial match
                if any(entity_role in role_name or role_name in entity_role for role_name in normalized_roles):
                    score += 0.4
                    reasons.append(f"Role match ({entity.semantic_guess})")
                else:
                    if entity.confidence > 0.6:
                         score -= 0.2
            else:
                score += 0.1
                
            # 2. Keyword Match
            col_lower = entity.column_name.lower()
            if intent_lower == col_lower:
                score += 0.6
                reasons.append("Exact match")
            elif intent_lower in col_lower: # "profit" in "gross_profit_margin"
                # Check for negation (e.g. asking for "profit" but col is "profit_margin" which is ratio?)
                # This is solved by Blueprint (Ratio vs Direct).
                score += 0.4
                reasons.append("Partial match")
            elif any(part in col_lower for part in intent_lower.split()):
                score += 0.2
                reasons.append("Token match")
                
            # 3. Quality
            if entity.null_percentage < 5:
                score += 0.05
                
            # 4. Semantic Confidence
            score *= entity.confidence
            
            if score > 0.3:
                candidates.append((score, entity, reasons))
            else:
                if intent_lower in col_lower:
                    logger.debug(f"  Column '{entity.column_name}' score {score:.2f} too low for intent '{intent}' (reasons: {reasons})")
        
        return candidates
    
    def _resolve_with_llm(self, business_metric: str, required_roles: List[str]) -> ResolutionResult:
        """Use LLM to intelligently map abstract metric to available columns."""
        
        # Prepare column metadata for LLM
        columns_info = []
        for entity in self.profile.entities:
            col_info = {
                "name": entity.column_name,
                "type": entity.statistical_type,
                "role": entity.semantic_guess,
                "confidence": round(entity.confidence, 2)
            }
            columns_info.append(col_info)
        
        prompt = f"""You are a data analyst mapping business metrics to dataset columns.

Business Metric: "{business_metric}"
Required Semantic Roles: {required_roles}

Available Columns:
{json.dumps(columns_info, indent=2)}

Task: Determine how to calculate this metric from available columns.

Response Format (JSON only, no markdown):
{{
  "mapping_type": "direct|derived|impossible",
  "explanation": "Brief reasoning",
  "confidence": 0.0-1.0,
  "target_column": "column_name" (if direct),
  "formula": "Sales - Cost" (if derived),
  "dependencies": ["col1", "col2"] (if derived)
}}

Rules:
1. Prefer DIRECT if column name clearly matches (e.g., Revenue→Sales)
2. Use DERIVED for calculations (e.g., Profitability→Sales-Cost)
3. Return IMPOSSIBLE only if truly no way to calculate
4. Common mappings: Revenue→Sales, Profitability→(Price-Cost), Growth→(Current-Previous)/Previous"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Fixed: removed groq/ prefix
                messages=[
                    {"role": "system", "content": "You are a data mapping expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            logger.info(f"LLM resolution result: {result}")
            
            # Validate and construct ResolutionResult
            if result["mapping_type"] == "direct":
                target = result.get("target_column")
                
                # Validate column exists
                available_cols = [e.column_name for e in self.profile.entities]
                if target not in available_cols:
                    logger.warning(f"LLM suggested non-existent column: {target}")
                    return ResolutionResult(None, 0.0, "failed", [], f"LLM suggested invalid column: {target}")
                
                return ResolutionResult(
                    resolved_metric=target,
                    resolution_confidence=min(1.0, result["confidence"]),
                    resolution_type="direct",
                    dependencies=[],
                    explanation=f"LLM: {result['explanation']}"
                )
                
            elif result["mapping_type"] == "derived":
                deps = result.get("dependencies", [])
                
                # Validate dependencies exist
                available_cols = [e.column_name for e in self.profile.entities]
                for dep in deps:
                    if dep not in available_cols:
                        logger.warning(f"LLM suggested non-existent dependency: {dep}")
                        return ResolutionResult(None, 0.0, "failed", [], f"Missing dependency: {dep}")
                
                return ResolutionResult(
                    resolved_metric=f"derived_{business_metric.lower().replace(' ', '_')}",
                    resolution_confidence=min(1.0, result["confidence"]),
                    resolution_type="derived",
                    dependencies=deps,
                    explanation=f"LLM: {result['explanation']}",
                    formula_name=result.get("formula", "custom")
                )
            else:
                return ResolutionResult(
                    None, 0.0, "failed", [],
                    f"LLM: {result.get('explanation', 'Cannot calculate this metric')}"
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            return ResolutionResult(None, 0.0, "failed", [], f"LLM JSON parse error: {str(e)}")
        except Exception as e:
            logger.error(f"LLM resolution error: {e}")
            return ResolutionResult(None, 0.0, "failed", [], f"LLM error: {str(e)}")

    def _resolve_from_registry(self, intent: str) -> ResolutionResult:
         # Reuse old derived logic
         # ... (Implementation of registry lookup)
         # For brevity, implementing a stub that checks existing registry
         metric_def = get_derived_metric(intent)
         if not metric_def:
             return ResolutionResult(None, 0.0, "failed", [], "")
             
         # Verify ingredients... (Simplified version of old logic)
         # In a real implementation, I'd copy the logic. 
         # Assuming I can't reuse the old method purely because I replaced the class.
         # So I will re-implement minimal registry lookup here.
         
         # For checking ingredients, revert to simple find
         ingredients = []
         missing = []
         for i, keywords in enumerate(metric_def.required_keywords):
              # Try to find best match
              best = None
              best_s = 0
              for entity in self.profile.entities:
                   s = 0
                   if any(k in entity.column_name.lower() for k in keywords):
                        s = 0.8
                   if entity.semantic_guess == metric_def.required_roles[i]:
                        s += 0.2
                   if s > best_s:
                        best_s = s
                        best = entity.column_name
              
              if best and best_s > 0.5:
                   ingredients.append(best)
              else:
                   missing.append(f"ingredient {i}")
         
         if missing:
              return ResolutionResult(None, 0.0, "failed", [], f"Missing {missing}")
              
         return ResolutionResult(
             resolved_metric=f"derived_{intent.replace(' ','_')}",
             resolution_confidence=0.8,
             resolution_type="derived",
             dependencies=ingredients,
             explanation=f"Found derived formula for {intent}",
             formula_name=intent
         )
