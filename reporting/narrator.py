"""
Insight Narrator - Converts statistical results into business narratives.

Uses Groq LLM to generate 4-section narratives (What/Why/So What/Now What).
"""
import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from groq import Groq
from utils.schemas import ValidatedInsight, BusinessContext, NarratedInsight
from utils.helpers import get_logger, to_json, generate_id

load_dotenv()
logger = get_logger(__name__)


class InsightNarrator:
    """Generates business narratives from validated insights using Groq LLM."""
    
    def __init__(self, token_tracker=None):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.token_tracker = token_tracker
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set")
        
        self.client = Groq(api_key=self.api_key)
    
    def narrate_insight(
        self,
        validated_insight: ValidatedInsight,
        business_context: BusinessContext
    ) -> NarratedInsight:
        """
        Convert validated insight into narrative form.
        
        Improvements:
        - Derives confidence from validator (Improvement #1)
        - Verdict-aware narration tone (Improvement #2)
        - Hard rule against causal claims (Improvement #3)
        - Requires evidence anchors (Improvement #4)
        
        Args:
            validated_insight: Validated insight with statistical results
            business_context: Business context for framing
            
        Returns:
            NarratedInsight with executive summaries
        """
        logger.info(f"Narrating insight: {validated_insight.hypothesis_id}")
        
        # Improvement #2: Check verdict - reject insights shouldn't be narrated
        verdict = validated_insight.validation.verdict
        if verdict == "reject":
            logger.warning(f"Skipping narration for rejected insight: {validated_insight.hypothesis_id}")
            return None  # Caller should filter these out
        
        # Extract key statistics
        result_data = validated_insight.result_data
        confidence_score = validated_insight.validation.confidence_score
        
        # Improvement #1: Derive narrative confidence from validator (NOT from LLM)
        if confidence_score >= 0.75:
            narrative_confidence = "high"
        elif confidence_score >= 0.4:
            narrative_confidence = "medium"
        else:
            narrative_confidence = "low"
        
        # Improvement #2: Verdict-aware narration instructions
        if verdict == "accept":
            tone_instruction = "Use assertive but factual language. This is a statistically strong finding."
        elif verdict == "weak_signal":
            tone_instruction = "Use exploratory language ('may indicate', 'early signal suggests', 'preliminary analysis shows'). This is an interesting pattern but not yet conclusive."
        else:
            tone_instruction = "Default factual tone."
        
        # Improvement #4: Prepare evidence anchors from result_data
        evidence_summary = self._extract_evidence_anchors(result_data)
        
        prompt = f"""You are a senior data consultant writing an executive insight for a CEO.

Business Context:
- Industry: {business_context.industry}
- Business Type: {business_context.business_type}
- Top KPIs: {', '.join(business_context.top_kpis)}

Insight Details:
- Lens: {validated_insight.lens}
- Finding: {validated_insight.title}
- Metric Analyzed: {validated_insight.metric}

Statistical Results:
{to_json(result_data, indent=2)[:800]}

Key Evidence:
{evidence_summary}

Validation Status:
- Verdict: {verdict}
- Confidence Score: {confidence_score:.2%}
- Derived Confidence Level: {narrative_confidence}

Provide a concise executive insight in JSON format:
{{
    "key_finding": "One sentence - what was discovered",
    "why_it_matters": "Why this is important for the business",
    "what_caused_it": "Likely driver or pattern (if identifiable from data)",
    "recommendation": "One clear actionable recommendation"
}}

CRITICAL RULES:
{tone_instruction}

IMPROVEMENT #3 - CAUSAL LANGUAGE RESTRICTION:
- If the analysis is correlational or observational, DO NOT claim causation
- Use phrases like "is associated with", "correlates with", "shows a relationship with"
- ONLY use causal language ("causes", "drives", "leads to") if the analysis explicitly used causal inference methods
- Analysis type hint: {result_data.get('analysis_type', 'correlation/trend')}

IMPROVEMENT #4 - EVIDENCE ANCHORING (MANDATORY):
- Include at least ONE concrete statistic (p-value, correlation, % change, effect size, confidence interval) in your "key_finding"
- Example: "Revenue increased by 23% (p<0.01)" or "Ad spend shows strong correlation with sales (r=0.78)"
- DO NOT make claims without numerical support

TONE RULES:
- Be honest and conservative
- Don't oversell weak correlations
- Acknowledge uncertainty for weak signals
- Focus on actionable insights
- Use executive language (concise, impactful)"""

        try:
            # Use Groq API for reliable narration
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert business consultant with deep analytical skills and statistical rigor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=600
            )
            
            content = response.choices[0].message.content
            
            # Track tokens if tracker available
            if self.token_tracker:
                self.token_tracker.record_call(
                    phase="narration",
                    purpose=f"Generate narrative for: {validated_insight.title}",
                    model=f"groq/{self.model}",
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            # Extract JSON
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            narration_data = json.loads(content)
            
            # Improvement #1: Use validator-derived confidence, NOT LLM's confidence
            narrated_insight = NarratedInsight(
                insight_id=generate_id("insight"),
                lens=validated_insight.lens,
                title=validated_insight.title,
                key_finding=narration_data["key_finding"],
                why_it_matters=narration_data["why_it_matters"],
                what_caused_it=narration_data.get("what_caused_it"),
                recommendation=narration_data["recommendation"],
                supporting_data=result_data,
                confidence=narrative_confidence,  # Derived from validator, not LLM
                verdict=verdict,  # Add verdict for downstream filtering
                resolution_type=validated_insight.resolution_type,
                formula_name=validated_insight.formula_name
            )
            
            logger.info(f"✓ Insight narrated: {narrated_insight.insight_id} (verdict: {verdict}, confidence: {narrative_confidence})")
            return narrated_insight
            
        except Exception as e:
            logger.error(f"Narration failed: {e}")
            
            # Fallback narration
            return NarratedInsight(
                insight_id=generate_id("insight"),
                lens=validated_insight.lens,
                title=validated_insight.title,
                key_finding=f"Analysis of {validated_insight.metric} revealed statistical patterns",
                why_it_matters="This metric shows variation that warrants attention",
                what_caused_it=None,
                recommendation="Further investigation recommended",
                supporting_data=result_data,
                confidence="low",
                verdict=verdict,
                resolution_type=validated_insight.resolution_type,
                formula_name=validated_insight.formula_name
            )
    
    def _extract_evidence_anchors(self, result_data: dict) -> str:
        """
        Improvement #4: Extract key statistics for evidence anchoring.
        
        Returns a formatted summary of key evidence.
        """
        evidence = []
        
        # P-value
        p_value = result_data.get("p_value") or result_data.get("pvalue")
        if p_value is not None:
            evidence.append(f"p-value: {p_value:.4f}")
        
        # Correlation
        correlation = result_data.get("correlation") or result_data.get("corr")
        if correlation is not None:
            evidence.append(f"correlation: {correlation:.3f}")
        
        # Effect size
        effect_size = result_data.get("effect_size") or result_data.get("estimated_effect")
        if effect_size is not None:
            evidence.append(f"effect size: {effect_size:.3f}")
        
        # Confidence interval
        conf_int = result_data.get("confidence_interval")
        if conf_int and isinstance(conf_int, dict):
            lower = conf_int.get("lower")
            upper = conf_int.get("upper")
            if lower is not None and upper is not None:
                evidence.append(f"95% CI: [{lower:.2f}, {upper:.2f}]")
        
        # Sample size
        sample_size = result_data.get("sample_size")
        if sample_size:
            evidence.append(f"n={sample_size}")
        
        # Forecast metrics
        model_confidence = result_data.get("model_confidence")
        if model_confidence is not None:
            evidence.append(f"model confidence: {model_confidence:.2f}")
        
        # Anomaly metrics
        anomalies_found = result_data.get("anomalies_found")
        if anomalies_found is not None:
            anomaly_rate = result_data.get("anomaly_rate", 0)
            evidence.append(f"{anomalies_found} anomalies ({anomaly_rate:.1%})")
        
        return ", ".join(evidence) if evidence else "No explicit statistics available"
    
    def generate_executive_summary(
        self,
        insights: List[NarratedInsight],
        business_context: BusinessContext
    ) -> str:
        """
        Generate overall executive summary.
        
        Improvement #5: Confidence-weighted summary.
        - Sorts by confidence
        - Filters out low-confidence noise
        
        Args:
            insights: List of all narrated insights
            business_context: Business context
            
        Returns:
            Executive summary text
        """
        logger.info(f"Generating executive summary for {len(insights)} insights")
        
        # Improvement #5: Sort by confidence and filter
        # Define confidence ordering
        confidence_order = {"high": 3, "medium": 2, "low": 1}
        
        # Sort insights by confidence (high → medium → low)
        sorted_insights = sorted(
            insights, 
            key=lambda x: confidence_order.get(x.confidence, 0), 
            reverse=True
        )
        
        # Filter: Keep high and medium, drop low unless we have very few insights
        if len(sorted_insights) >= 5:
            # Drop low-confidence insights
            filtered_insights = [i for i in sorted_insights if i.confidence in ["high", "medium"]]
        else:
            # Keep all if we have few insights
            filtered_insights = sorted_insights
        
        logger.info(f"Filtered insights: {len(filtered_insights)} (from {len(insights)} total)")
        
        # Group insights by lens
        insights_by_lens = {}
        for insight in filtered_insights:
            if insight.lens not in insights_by_lens:
                insights_by_lens[insight.lens] = []
            insights_by_lens[insight.lens].append(insight)
        
        # Count by confidence
        high_confidence = len([i for i in filtered_insights if i.confidence == "high"])
        medium_confidence = len([i for i in filtered_insights if i.confidence == "medium"])
        
        prompt = f"""You are a CEO advisor. Synthesize these insights into a brief executive summary.

Business: {business_context.industry} - {business_context.business_type}
Total Insights Analyzed: {len(insights)}
Selected for Summary: {len(filtered_insights)}
- High Confidence: {high_confidence}
- Medium Confidence: {medium_confidence}

Insights by Perspective (sorted by confidence):
"""
        
        for lens, lens_insights in insights_by_lens.items():
            prompt += f"\n{lens} Perspective ({len(lens_insights)} insights):\n"
            for insight in lens_insights[:3]:  # Top 3 per lens
                prompt += f"[{insight.confidence.upper()}] {insight.key_finding}\n"
        
        prompt += """

Write a 3-4 sentence executive summary that:
1. Highlights the most critical findings (prioritize high-confidence insights)
2. Emphasizes business impact
3. Suggests strategic direction

RULES:
- Be concise, honest, and actionable
- Focus on high-confidence findings
- Acknowledge if insights are preliminary (medium confidence)
- Do not make causal claims unless explicitly supported"""

        try:
            # Use Groq API for executive summary (switched from HuggingFace)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic business advisor to C-level executives."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Track tokens if tracker available
            if self.token_tracker:
                self.token_tracker.record_call(
                    phase="executive_summary",
                    purpose="Generate executive summary",
                    model=f"groq/{self.model}",
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            logger.info("Executive summary generated (confidence-weighted)")
            return summary
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return f"Analysis of {business_context.business_type} data revealed {len(filtered_insights)} statistically validated insights across financial, operational, and marketing dimensions. {high_confidence} insights show high confidence and warrant immediate attention."
