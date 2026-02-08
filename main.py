"""
Main pipeline orchestrator for the Autonomous Insight Engine.

Coordinates all phases: ingestion, profiling, hypothesis generation,
metric resolution, execution planning, validation, and reporting.
"""
import os
import polars as pl
from typing import Optional, List
from datetime import datetime

from core.robust_loader import RobustLoader
from core.profiling import SemanticProfiler
from core.context import ContextInjector
from core.data_health import DataHealthChecker
from core.grain_discovery import GrainDiscoverer
from intelligence.lenses import CouncilOfLenses
from core.metric_resolver import MetricResolver
from intelligence.planner import ExecutionPlanner
from execution.self_healing import SelfHealingExecutor
from validation.adversary import AdversarialValidator
from reporting.narrator import InsightNarrator
from reporting.report_builder import ReportBuilder

from utils.schemas import (
    PipelineProgress,
    AnalysisReport,
    NarratedInsight
)
from utils.helpers import get_logger
from utils.token_tracker import TokenTracker

logger = get_logger(__name__)


class AutonomousInsightEngine:
    """Main pipeline orchestrator."""
    
    def __init__(self):
        # self.ingestor = None # Deprecated
        self.profiler = None
        self.df: Optional[pl.DataFrame] = None
        self.token_tracker = None  # Will be initialized per run
        self.progress = PipelineProgress(
            phase="ingest",
            progress_percentage=0.0,
            current_step="Initializing"
        )
    
    def run_pipeline(self, csv_path: str, dataset_name: str = "Dataset") -> AnalysisReport:
        """
        Run the complete autonomous insight pipeline.
        
        Args:
            csv_path: Path to CSV file
            dataset_name: Name of the dataset for reporting
            
        Returns:
            AnalysisReport with all insights and metadata
        """
        logger.info("=" * 60)
        logger.info("AUTONOMOUS INSIGHT ENGINE - PIPELINE START")
        logger.info("=" * 60)
        
        # PHASE 0: Ingest & Clean
        self._update_progress("ingest", 5, "Loading and cleaning data")
        logger.info("\n[PHASE 0] ROBUST INGEST & CLEAN")
        
        # Improvement: Use RobustLoader
        loader = RobustLoader(csv_path)
        self.df = loader.load_and_clean()
        
        logger.info(f"✓ Data loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        # Initialize token tracker
        self.token_tracker = TokenTracker()
        logger.info("✓ Token tracker initialized")
        
        # Initialize LLM client (Groq)
        from groq import Groq
        self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # PHASE 0.1: Data Health Check (NEW)
        self._update_progress("health_check", 10, "Validating data quality")
        logger.info("\n[PHASE 0.1] DATA HEALTH CHECK")
        
        health_checker = DataHealthChecker(self.df, min_score=70.0)
        health_report = health_checker.check_health()
        
        if not health_report.should_proceed:
            logger.error(f"✗ Data quality insufficient: {health_report.overall_score:.1f}/100")
            for issue in health_report.critical_issues:
                logger.error(f"  - {issue}")
            
            # Return early with health report
            return AnalysisReport(
                dataset_name=dataset_name,
                timestamp=datetime.now().isoformat(),
                insights=[],
                metadata={
                    "error": "Data quality too low for reliable analysis",
                    "health_score": health_report.overall_score,
                    "critical_issues": health_report.critical_issues,
                    "warnings": health_report.warnings
                }
            )
        
        logger.info(f"✓ Data Health Score: {health_report.overall_score:.1f}/100")
        if health_report.warnings:
            logger.info(f"  Warnings: {len(health_report.warnings)}")
            for warning in health_report.warnings[:3]:  # Show first 3
                logger.info(f"    - {warning}")
        
        # PHASE 0.5: Context Injection
        self._update_progress("context", 15, "Inferring business context")
        logger.info("\n[PHASE 0.5] CONTEXT INJECTION (LLM #1 - Groq)")
        
        self.profiler = SemanticProfiler(self.df)
        semantic_profile = self.profiler.profile_dataset()
        
        context_injector = ContextInjector(token_tracker=self.token_tracker)
        business_context = context_injector.infer_context(semantic_profile)
        
        logger.info(f"✓ Context: {business_context.industry} - {business_context.business_type}")
        logger.info(f"  KPIs: {', '.join(business_context.top_kpis)}")
        
        # Store context in profile
        semantic_profile.business_context = business_context
        
        # PHASE 1: Semantic Profiling
        self._update_progress("profiling", 25, "Analyzing data structure")
        logger.info("\n[PHASE 1] SEMANTIC PROFILING")
        
        relationships = self.profiler.detect_relationships()
        logger.info(f"✓ Detected {len(relationships)} entity relationships")
        
        # PHASE 1.2: Grain Discovery (NEW)
        self._update_progress("grain_discovery", 30, "Discovering data grain")
        logger.info("\n[PHASE 1.2] GRAIN DISCOVERY")
        
        grain_discoverer = GrainDiscoverer(self.df, semantic_profile)
        grain_profile = grain_discoverer.discover_grain()
        semantic_profile.grain = grain_profile
        
        logger.info(f"✓ Grain: {grain_profile.row_represents}")
        if grain_profile.aggregation_needed:
            logger.info(f"  → Aggregation required for time series analysis")
        
        # PHASE 2: Council of Lenses
        self._update_progress("hypotheses", 35, "Generating hypotheses")
        logger.info("\n[PHASE 2] COUNCIL OF LENSES (LLM #1 - Groq)")
        
        council = CouncilOfLenses()
        hypotheses = council.generate_all_hypotheses(business_context, semantic_profile)
        
        logger.info(f"✓ Generated {len(hypotheses)} raw hypotheses")
        
        # PHASE 2.5: Metric Resolution
        self._update_progress("resolution", 40, "Resolving abstract metrics")
        logger.info("\n[PHASE 2.5] METRIC RESOLUTION")
        
        metric_resolver = MetricResolver(semantic_profile, llm_client=self.llm_client)
        resolved_hypotheses = []
        
        for h in hypotheses:
            resolution = metric_resolver.resolve_metric(h.business_metric, h.required_semantic_roles)
            
            if resolution.resolution_type in ["failed", "unsupported"]:
                logger.warning(f"✗ Dropping '{h.title}': {resolution.explanation}")
                continue
            
            # Update hypothesis with resolution details
            h.resolved_metric = resolution.resolved_metric
            h.resolution_confidence = resolution.resolution_confidence
            h.resolution_type = resolution.resolution_type
            h.formula_name = resolution.formula_name
            h.dependencies = resolution.dependencies
            
            logger.info(f"✓ Resolved '{h.business_metric}' -> '{h.resolved_metric}' ({resolution.resolution_type})")
            resolved_hypotheses.append(h)
            
        logger.info(f"✓ Proceeding with {len(resolved_hypotheses)}/{len(hypotheses)} validated hypotheses")
        
        # PHASE 3: Execution Planning
        self._update_progress("planning", 45, "Planning computations")
        logger.info("\n[PHASE 3] EXECUTION PLANNING")
        
        planner = ExecutionPlanner(self.df, semantic_profile)
        execution_plan = planner.create_plan(resolved_hypotheses)
        
        logger.info(f"✓ Execution plan ready: {len(execution_plan.hypotheses)} hypotheses")
        
        # PHASE 4: Self-Healing Execution
        self._update_progress("execution", 55, "Executing analyses")
        logger.info("\n[PHASE 4] SELF-HEALING EXECUTION")
        
        executor = SelfHealingExecutor(self.df)
        computation_results = []
        
        for i, hypothesis in enumerate(execution_plan.hypotheses):
            logger.info(f"\n[{i+1}/{len(execution_plan.hypotheses)}] Executing: {hypothesis.title}")
            result = executor.execute_hypothesis(hypothesis)
            computation_results.append((hypothesis, result))
        
        successful_executions = [r for _, r in computation_results if r.success]
        logger.info(f"\n✓ Execution complete: {len(successful_executions)}/{len(computation_results)} successful")
        
        # PHASE 5: Adversarial Validation
        self._update_progress("validation", 70, "Validating insights")
        logger.info("\n[PHASE 5] ADVERSARIAL VALIDATION")
        
        validator = AdversarialValidator(self.df)
        validated_insights = []
        
        for hypothesis, result in computation_results:
            if result.success:
                validation = validator.validate_result(hypothesis, result)
                
                if validation.passed:
                    validated_insight = validator.create_validated_insight(hypothesis, result, validation)
                    validated_insights.append(validated_insight)
        
        logger.info(f"✓ Validation complete: {len(validated_insights)}/{len(successful_executions)} insights passed")
        rejection_count = len(successful_executions) - len(validated_insights)
        
        # PHASE 6: Narration
        self._update_progress("narration", 85, "Generating narratives")
        logger.info("\n[PHASE 6] INSIGHT NARRATION")
        
        narrator = InsightNarrator(token_tracker=self.token_tracker)
        narrated_insights = []
        
        for validated_insight in validated_insights:
            try:
                narrative = narrator.narrate_insight(validated_insight, business_context)
                if narrative:  # narrate_insight returns None for rejected insights
                    narrated_insights.append(narrative)
                    logger.info(f"✓ Narrated: {validated_insight.title}")
            except Exception as e:
                logger.warning(f"⚠ Narration failed for '{validated_insight.title}': {e}")
                # Create fallback NarratedInsight
                from utils.schemas import NarratedInsight
                from utils.helpers import generate_id
                
                # Derive confidence from validation score
                conf_score = validated_insight.validation.confidence_score
                if conf_score >= 0.75:
                    confidence = "high"
                elif conf_score >= 0.4:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                fallback_narrative = NarratedInsight(
                    insight_id=generate_id("insight"),
                    lens=validated_insight.lens,
                    title=validated_insight.title,
                    key_finding=f"Analysis of {validated_insight.metric}",
                    why_it_matters=f"Statistical analysis of {validated_insight.title}",
                    recommendation="Review the statistical results for details.",
                    supporting_data=validated_insight.result_data,
                    confidence=confidence,
                    verdict=validated_insight.validation.verdict,
                    resolution_type=validated_insight.resolution_type,
                    formula_name=validated_insight.formula_name
                )
                narrated_insights.append(fallback_narrative)
                logger.info(f"✓ Used fallback narrative for: {validated_insight.title}")
        
        logger.info(f"✓ Narration complete: {len(narrated_insights)} insights narrated")
        
        # Generate executive summary
        executive_summary = narrator.generate_executive_summary(narrated_insights, business_context)
        
        # PHASE 7: Report Generation
        self._update_progress("reporting", 90, "Building report")
        logger.info("\n[PHASE 7] REPORT GENERATION")
        
        # Create analysis report
        analysis_report = AnalysisReport(
            dataset_name=dataset_name,
            analysis_timestamp=datetime.now(),
            business_context=business_context,
            insights=narrated_insights,
            charts=[],
            executive_summary=executive_summary,
            total_hypotheses=len(hypotheses),
            validated_insights=len(validated_insights),
            rejection_count=rejection_count
        )
        
        # Build HTML report
        report_builder = ReportBuilder()
        html_path = report_builder.build_report(analysis_report)
        logger.info(f"✓ HTML report: {html_path}")
        
        # Convert to PDF (optional - may fail on Windows due to weasyprint/fontconfig)
        try:
            pdf_path = report_builder.convert_to_pdf(html_path)
            if pdf_path:
                logger.info(f"✓ PDF report: {pdf_path}")
        except Exception as e:
            logger.warning(f"⚠ PDF generation skipped (weasyprint/fontconfig issue): {str(e)[:100]}")
            logger.info("✓ HTML report available, PDF generation disabled")
        
        # COMPLETE
        self._update_progress("complete", 100, "Analysis complete")
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Hypotheses: {len(hypotheses)}")
        logger.info(f"Validated Insights: {len(validated_insights)}")
        logger.info(f"Rejected: {rejection_count}")
        logger.info("=" * 60)
        
        return analysis_report
    
    def _update_progress(self, phase: str, percentage: float, step: str):
        """Update pipeline progress."""
        self.progress.phase = phase
        self.progress.progress_percentage = percentage
        self.progress.current_step = step
        logger.info(f"[{percentage}%] {step}")
    
    def get_progress(self) -> PipelineProgress:
        """Get current pipeline progress."""
        return self.progress


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_file_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    engine = AutonomousInsightEngine()
    report = engine.run_pipeline(csv_path)
    
    print(f"\n✓ Analysis complete!")
    print(f"  Insights: {len(report.insights)}")
    print(f"  Report generated in: reports/")
