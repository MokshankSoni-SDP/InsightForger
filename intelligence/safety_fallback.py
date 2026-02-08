"""
Safety Fallback Wrapper for Phase 1 & Phase 2 Integration

This module provides a safety mechanism that retries Phase 1 hypothesis generation
if Phase 2 rejects more than 50% of hypotheses due to hallucinated columns.
"""
from typing import List, Tuple, Dict
from utils.schemas import Hypothesis, BusinessContext, SemanticProfile
from intelligence.lenses import CouncilOfLenses
from core.metric_resolver import MetricResolver
from utils.helpers import get_logger
import pandas as pd

logger = get_logger(__name__)


def generate_hypotheses_with_fallback(
    business_context: BusinessContext,
    profile: SemanticProfile,
    df: pd.DataFrame,
    metric_resolver: MetricResolver
) -> Tuple[List[Hypothesis], Dict[str, int]]:
    """
    Generate hypotheses with automatic retry if >50% are rejected by Phase 2.
    
    Args:
        business_context: Business context from Phase 0.5
        profile: Semantic profile
        df: DataFrame for sampling
        metric_resolver: Metric resolver for validation
        
    Returns:
        Tuple of (validated_hypotheses, stats_dict)
    """
    council = CouncilOfLenses()
    
    # First attempt
    logger.info("=" * 60)
    logger.info("PHASE 1: HYPOTHESIS GENERATION (Attempt 1)")
    logger.info("=" * 60)
    
    hypotheses = council.generate_all_hypotheses(business_context, profile, df)
    
    # Validate with Phase 2
    logger.info(f"\nValidating {len(hypotheses)} hypotheses with Phase 2...")
    
    validated = []
    rejected = []
    rejection_reasons = {}
    
    for hyp in hypotheses:
        try:
            # Try to resolve the hypothesis
            execution_plan = metric_resolver.resolve_hypothesis(hyp)
            validated.append(hyp)
            logger.info(f"✓ {hyp.title}")
        except Exception as e:
            rejected.append(hyp)
            error_msg = str(e)
            logger.warning(f"✗ {hyp.title}: {error_msg}")
            
            # Track rejection reasons
            if "missing from CSV" in error_msg:
                # Extract missing columns
                import re
                match = re.search(r"missing from CSV: \[(.*?)\]", error_msg)
                if match:
                    missing_cols = match.group(1).replace("'", "").split(", ")
                    for col in missing_cols:
                        rejection_reasons[col] = rejection_reasons.get(col, 0) + 1
    
    rejection_rate = len(rejected) / len(hypotheses) if hypotheses else 0
    
    logger.info(f"\n📊 Validation Results:")
    logger.info(f"   ✓ Validated: {len(validated)}/{len(hypotheses)} ({(1-rejection_rate)*100:.1f}%)")
    logger.info(f"   ✗ Rejected: {len(rejected)}/{len(hypotheses)} ({rejection_rate*100:.1f}%)")
    
    # SAFETY FALLBACK: Retry if >50% rejected
    if rejection_rate > 0.5 and rejection_reasons:
        logger.warning(f"\n⚠️ SAFETY FALLBACK TRIGGERED: {rejection_rate*100:.1f}% rejection rate")
        logger.info("Retrying Phase 1 with correction notes...")
        
        # Build correction notes
        correction_notes = "HALLUCINATED COLUMNS (these don't exist in the CSV):\n"
        for col, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            correction_notes += f"- '{col}' (used in {count} hypotheses - DO NOT USE!)\n"
        correction_notes += "\nPlease use ONLY columns from the EXACT_KEY list in AVAILABLE COLUMNS!"
        
        logger.info(f"\nCorrection Notes:\n{correction_notes}")
        
        # Retry each lens with corrections
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: HYPOTHESIS GENERATION (Attempt 2 - WITH CORRECTIONS)")
        logger.info("=" * 60)
        
        retried_hypotheses = []
        for lens_rec in business_context.recommended_lenses:
            from intelligence.lenses import LensAgent
            
            agent = LensAgent(
                lens_recommendation=lens_rec,
                business_context=business_context,
                profile=profile,
                guardrails=business_context.guardrails,
                df=df
            )
            
            # Retry with corrections
            corrected_hyps = agent.retry_with_corrections(correction_notes)
            retried_hypotheses.extend(corrected_hyps)
        
        # Validate retried hypotheses
        logger.info(f"\nValidating {len(retried_hypotheses)} retried hypotheses...")
        
        validated_retry = []
        rejected_retry = []
        
        for hyp in retried_hypotheses:
            try:
                execution_plan = metric_resolver.resolve_hypothesis(hyp)
                validated_retry.append(hyp)
                logger.info(f"✓ {hyp.title}")
            except Exception as e:
                rejected_retry.append(hyp)
                logger.warning(f"✗ {hyp.title}: {str(e)}")
        
        retry_rejection_rate = len(rejected_retry) / len(retried_hypotheses) if retried_hypotheses else 0
        
        logger.info(f"\n📊 Retry Results:")
        logger.info(f"   ✓ Validated: {len(validated_retry)}/{len(retried_hypotheses)} ({(1-retry_rejection_rate)*100:.1f}%)")
        logger.info(f"   ✗ Rejected: {len(rejected_retry)}/{len(retried_hypotheses)} ({retry_rejection_rate*100:.1f}%)")
        
        # Use retried hypotheses if better
        if len(validated_retry) > len(validated):
            logger.info(f"\n✅ Retry improved results: {len(validated_retry)} vs {len(validated)} validated")
            validated = validated_retry
        else:
            logger.info(f"\n⚠️ Retry did not improve results, keeping original: {len(validated)} validated")
    
    stats = {
        "total_generated": len(hypotheses),
        "validated": len(validated),
        "rejected": len(rejected),
        "rejection_rate": rejection_rate,
        "retry_triggered": rejection_rate > 0.5
    }
    
    return validated, stats
