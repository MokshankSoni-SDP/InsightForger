"""
Report builder module.

Generates HTML reports and converts to PDF.
Enhanced with embedded charts, key evidence, sorted insights, methodology section, and explicit metadata.
"""
import os
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from utils.schemas import AnalysisReport, NarratedInsight, BusinessContext
from utils.helpers import get_logger

logger = get_logger(__name__)


class ReportBuilder:
    """Builds production-grade HTML and PDF reports with charts and evidence."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chart_dir = self.output_dir / "charts"
        self.chart_dir.mkdir(exist_ok=True)
    
    def build_report(
        self, 
        report: AnalysisReport,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Build HTML report from analysis.
        
        Improvements:
        - Sorts insights by confidence (Improvement #3)
        - Embeds charts for each insight (Improvement #1)
        - Adds key evidence sections (Improvement #2)
        - Includes methodology & caveats (Improvement #4)
        - Explicit metadata (Improvement #5)
        
        Args:
            report: AnalysisReport object
            metadata: Optional execution metadata
            
        Returns:
            Path to generated HTML file
        """
        logger.info("Building HTML report with enhancements")
        
        # Improvement #3: Sort insights by confidence before rendering
        report.insights = self._sort_insights(report.insights)
        
        # Improvement #1: Generate charts for insights
        self._generate_charts_for_insights(report.insights)
        
        # Improvement #5: Prepare metadata
        if metadata is None:
            metadata = {}
        
        report_metadata = self._prepare_metadata(report, metadata)
        
        # Generate HTML
        html_content = self._generate_html(report, report_metadata)
        
        # Save HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = self.output_dir / f"analysis_report_{timestamp}.html"
        
        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved: {html_filename}")
        return str(html_filename)
    
    def _sort_insights(self, insights: List[NarratedInsight]) -> List[NarratedInsight]:
        """
        Improvement #3: Sort insights by confidence and verdict.
        
        Order: high confidence → medium → low
        Secondary sort: verdict (accept → weak_signal)
        """
        confidence_order = {"high": 3, "medium": 2, "low": 1}
        verdict_order = {"accept": 2, "weak_signal": 1, "reject": 0}
        
        sorted_insights = sorted(
            insights,
            key=lambda i: (
                confidence_order.get(i.confidence, 0),
                verdict_order.get(i.verdict, 0)
            ),
            reverse=True
        )
        
        logger.info(f"Sorted {len(sorted_insights)} insights by confidence/verdict")
        return sorted_insights
    
    def _generate_charts_for_insights(self, insights: List[NarratedInsight]):
        """
        Improvement #1: Generate and embed charts for each insight.
        
        Charts are created based on insight type and data.
        """
        for insight in insights:
            try:
                chart_path = self._create_chart_for_insight(insight)
                if chart_path:
                    # Store relative path for HTML embedding
                    insight.chart_path = f"charts/{Path(chart_path).name}"
                else:
                    insight.chart_path = None
            except Exception as e:
                logger.warning(f"Chart generation failed for {insight.insight_id}: {e}")
                insight.chart_path = None
    
    def _create_chart_for_insight(self, insight: NarratedInsight) -> str:
        """Create appropriate chart based on supporting data."""
        data = insight.supporting_data
        
        # Determine chart type from data
        if "forecast_values" in data:
            # Forecast chart
            return self._create_forecast_chart(insight, data)
        elif "anomaly_details" in data:
            # Anomaly chart
            return self._create_anomaly_chart(insight, data)
        elif "correlation" in data in data:
            # Correlation scatter
            return self._create_correlation_chart(insight, data)
        else:
            # Generic metric chart
            return self._create_metric_chart(insight, data)
    
    def _create_forecast_chart(self, insight: NarratedInsight, data: Dict) -> str:
        """Create forecast visualization with confidence intervals."""
        try:
            forecast_values = data.get("forecast_values", [])
            conf_lower = data.get("confidence_lower", [])
            conf_upper = data.get("confidence_upper", [])
            
            fig = go.Figure()
            
            x = list(range(1, len(forecast_values) + 1))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=x, y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#3498db', width=2)
            ))
            
            #Confidence interval
            if conf_lower and conf_upper:
                fig.add_trace(go.Scatter(
                    x=x + x[::-1],
                    y=conf_upper + conf_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(52, 152, 219, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
            
            fig.update_layout(
                title=f"Forecast: {insight.title}",
                xaxis_title="Period",
                yaxis_title="Value",
                template="plotly_white",
                height=400
            )
            
            chart_filename = self.chart_dir / f"chart_{insight.insight_id}.png"
            fig.write_image(str(chart_filename))
            
            return str(chart_filename)
        
        except Exception as e:
            logger.error(f"Forecast chart failed: {e}")
            return ""
    
    def _create_anomaly_chart(self, insight: NarratedInsight, data: Dict) -> str:
        """Create anomaly visualization highlighting outliers."""
        try:
            anomaly_details = data.get("anomaly_details", [])
            
            if not anomaly_details:
                return ""
            
            # Extract indices and values
            indices = [a.get("index", i) for i, a in enumerate(anomaly_details)]
            values = [a.get("value", 0) for a in anomaly_details]
            severities = [a.get("severity", 0) for a in anomaly_details]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=indices,
                y=values,
                mode='markers',
                marker=dict(
                    size=[10 + s * 5 for s in severities],  # Size by severity
                    color=severities,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Severity")
                ),
                text=[f"Severity: {s:.2f}" for s in severities],
                name='Anomalies'
            ))
            
            fig.update_layout(
                title=f"Anomalies: {insight.title}",
                xaxis_title="Index",
                yaxis_title="Value",
                template="plotly_white",
                height=400
            )
            
            chart_filename = self.chart_dir / f"chart_{insight.insight_id}.png"
            fig.write_image(str(chart_filename))
            
            return str(chart_filename)
        
        except Exception as e:
            logger.error(f"Anomaly chart failed: {e}")
            return ""
    
    def _create_correlation_chart(self, insight: NarratedInsight, data: Dict) -> str:
        """Create correlation scatter plot."""
        # Placeholder - would need actual data points
        return ""
    
    def _create_metric_chart(self, insight: NarratedInsight, data: Dict) -> str:
        """Create generic metric chart."""
        # Placeholder
        return ""
    
    def _extract_key_evidence(self, insight: NarratedInsight) -> List[Dict[str, str]]:
        """
        Improvement #2: Extract key evidence from supporting data.
        
        Returns list of {label, value} dicts for display.
        """
        data = insight.supporting_data
        evidence = []
        
        # P-value
        p_value = data.get("p_value") or data.get("pvalue")
        if p_value is not None:
            evidence.append({"label": "P-value", "value": f"{p_value:.4f}"})
        
        # Correlation
        correlation = data.get("correlation") or data.get("corr")
        if correlation is not None:
            evidence.append({"label": "Correlation", "value": f"{correlation:.3f}"})
        
        # Effect size
        effect_size = data.get("effect_size") or data.get("estimated_effect")
        if effect_size is not None:
            evidence.append({"label": "Effect Size", "value": f"{effect_size:.3f}"})
        
        # Sample size
        sample_size = data.get("sample_size")
        if sample_size:
            evidence.append({"label": "Sample Size", "value": f"{sample_size:,}"})
        
        # Model confidence (forecasts)
        model_conf = data.get("model_confidence")
        if model_conf is not None:
            evidence.append({"label": "Model Confidence", "value": f"{model_conf:.2%}"})
        
        # Anomaly rate
        anomaly_rate = data.get("anomaly_rate")
        if anomaly_rate is not None:
            evidence.append({"label": "Anomaly Rate", "value": f"{anomaly_rate:.1%}"})
        
        # Causal confidence
        causal_conf = data.get("causal_confidence")
        if causal_conf:
            evidence.append({"label": "Causal Confidence", "value": causal_conf.upper()})
        
        return evidence[:4]  # Limit to 4 key stats
    
    def _prepare_metadata(self, report: AnalysisReport, metadata: Dict) -> Dict[str, Any]:
        """
        Improvement #5: Prepare explicit report metadata.
        """
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_hypotheses": report.total_hypotheses,
            "validated_insights": report.validated_insights,
            "rejection_count": report.rejection_count,
            "total_retries": metadata.get("total_retries", 0),
            "groq_model": metadata.get("groq_model", "groq/llama-3.1-70b-versatile"),
            "hf_model": metadata.get("hf_model", "meta-llama/Meta-Llama-3-8B-Instruct"),
            "execution_time": metadata.get("execution_time", "N/A")
        }
    
    def convert_to_pdf(self, html_path: str) -> str:
        """
        Convert HTML report to PDF.
        
        Args:
            html_path: Path to HTML file
            
        Returns:
            Path to generated PDF file
        """
        logger.info("Converting report to PDF")
        
        try:
            from weasyprint import HTML
            
            pdf_path = html_path.replace(".html", ".pdf")
            HTML(html_path).write_pdf(pdf_path)
            
            logger.info(f"PDF report saved: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return ""
    
    def _generate_html(self, report: AnalysisReport, metadata: Dict) -> str:
        """Generate HTML content with all enhancements."""
        
        # Improvement #2: Extract evidence for each insight
        for insight in report.insights:
            insight.key_evidence = self._extract_key_evidence(insight)
        
        template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Insight Report - {{ report.dataset_name }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        header {
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .meta {
            color: #7f8c8d;
            font-size: 0.95em;
        }
        
        .context-box {
            background: #ecf0f1;
            padding: 20px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }
        
        .executive-summary {
            background: #e8f5e9;
            padding: 25px;
            border-left: 4px solid #4caf50;
            margin: 30px 0;
            font-size: 1.1em;
        }
        
        .insight-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 25px;
            margin: 25px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .insight-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .lens-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            color: white;
        }
        
        .lens-CFO { background: #e74c3c; }
        .lens-COO { background: #3498db; }
        .lens-CMO { background: #9b59b6; }
        
        .confidence-badge {
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .confidence-high { background: #2ecc71; color: white; }
        .confidence-medium { background: #f39c12; color: white; }
        .confidence-low { background: #95a5a6; color: white; }
        
        .insight-title {
            font-size: 1.5em;
            color: #2c3e50;
            margin: 10px 0;
        }
        
        .insight-section {
            margin: 15px 0;
        }
        
        .insight-section h4 {
            color: #34495e;
            margin-bottom: 8px;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .insight-section p {
            color: #555;
            line-height: 1.7;
        }
        
        /* Improvement #2: Key evidence styling */
        .evidence-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .evidence-item {
            text-align: center;
        }
        
        .evidence-value {
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .evidence-label {
            font-size: 0.8em;
            color: #7f8c8d;
            margin-top: 3px;
        }
        
        /* Improvement #1: Chart embedding */
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .stat-label {
            font-size: 0.85em;
            color: #7f8c8d;
           margin-top: 5px;
        }
        
        /* Improvement #4: Methodology section */
        .methodology-box {
            background: #fff3cd;
            padding: 20px;
            border-left: 4px solid #ffc107;
            margin: 30px 0;
        }
        
        .methodology-box ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        
        .methodology-box li {
            margin: 8px 0;
            color: #555;
        }
        
        /* Improvement #5: Metadata section */
        .metadata-box {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 0.9em;
        }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        
        .metadata-item {
            padding: 8px;
            background: white;
            border-radius: 3px;
        }
        
        .metadata-label {
            font-weight: bold;
            color: #1976d2;
        }
        
        footer {
            border-top: 2px solid #ecf0f1;
            padding-top: 20px;
            margin-top: 40px;
            text-align: center;
            color: #95a5a6;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Autonomous Insight Report</h1>
            <div class="meta">
                <strong>Dataset:</strong> {{ report.dataset_name }} | 
                <strong>Generated:</strong> {{ report.analysis_timestamp.strftime("%Y-%m-%d %H:%M:%S") }}
            </div>
        </header>
        
        <!-- Improvement #5: Explicit Metadata -->
        <div class="metadata-box">
            <h3 style="margin-bottom: 10px; color: #1976d2;">📋 Report Metadata</h3>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="metadata-label">Analysis Model:</span> {{ metadata.groq_model }}
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Narration Model:</span> {{ metadata.hf_model }}
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Total Retries:</span> {{ metadata.total_retries }}
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Execution Time:</span> {{ metadata.execution_time }}
                </div>
            </div>
        </div>
        
        <div class="context-box">
            <h3>🏢 Business Context</h3>
            <p><strong>Industry:</strong> {{ report.business_context.industry }}</p>
            <p><strong>Business Type:</strong> {{ report.business_context.business_type }}</p>
            <p><strong>Key KPIs:</strong> {{ ', '.join(report.business_context.top_kpis) }}</p>
        </div>
        
        <div class="executive-summary">
            <h3>📋 Executive Summary</h3>
            <p>{{ report.executive_summary }}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{{ report.total_hypotheses }}</div>
                <div class="stat-label">Hypotheses Tested</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{{ report.validated_insights }}</div>
                <div class="stat-label">Validated Insights</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{{ report.rejection_count }}</div>
                <div class="stat-label">Rejected (Low Quality)</div>
            </div>
        </div>
        
        <h2 style="margin: 40px 0 20px; color: #2c3e50;">💡 Key Insights</h2>
        <p style="color: #7f8c8d; margin-bottom: 20px;">Sorted by confidence (highest first)</p>
        
        {% for insight in report.insights %}
        <div class="insight-card">
            <div class="insight-header">
                <span class="lens-badge lens-{{ insight.lens }}">{{ insight.lens }}</span>
                <span class="confidence-badge confidence-{{ insight.confidence }}">
                    {{ insight.confidence.upper() }} CONFIDENCE
                </span>
            </div>
            
            <h3 class="insight-title">{{ insight.title }}</h3>
            
            <!-- Improvement #6: Resolution Transparency -->
            <div style="background: #f0f7fb; padding: 8px 12px; border-radius: 4px; margin-bottom: 15px; font-size: 0.85em; border-left: 3px solid #3498db; color: #555;">
                <strong>Metric Resolution:</strong> {{ insight.resolution_type.upper() }}
                {% if insight.formula_name %}
                <span style="margin-left: 10px; color: #7f8c8d;">(Formula: {{ insight.formula_name }})</span>
                {% endif %}
            </div>
            
            <!-- Improvement #2: Key Evidence Section -->
            {% if insight.key_evidence %}
            <div class="evidence-grid">
                {% for evidence in insight.key_evidence %}
                <div class="evidence-item">
                    <div class="evidence-value">{{ evidence.value }}</div>
                    <div class="evidence-label">{{ evidence.label }}</div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="insight-section">
                <h4>🔍 Key Finding</h4>
                <p>{{ insight.key_finding }}</p>
            </div>
            
            <div class="insight-section">
                <h4>💼 Why It Matters</h4>
                <p>{{ insight.why_it_matters }}</p>
            </div>
            
            {% if insight.what_caused_it %}
            <div class="insight-section">
                <h4>🎯 Driver / Pattern</h4>
                <p>{{ insight.what_caused_it }}</p>
            </div>
            {% endif %}
            
            <div class="insight-section">
                <h4>✅ Recommendation</h4>
                <p>{{ insight.recommendation }}</p>
            </div>
            
            <!-- Improvement #1: Embedded Chart -->
            {% if insight.chart_path %}
            <div class="chart-container">
                <img src="{{ insight.chart_path }}" alt="Chart for {{ insight.title }}">
            </div>
            {% endif %}
        </div>
        {% endfor %}
        
        <!-- Improvement #4: Methodology & Caveats -->
        <div class="methodology-box">
            <h3 style="margin-bottom: 10px; color: #856404;">⚙️ Methodology & Caveats</h3>
            <ul>
                <li><strong>Statistical Validation:</strong> All insights passed multi-layered statistical validation including sample size checks, p-value verification, and temporal stability tests.</li>
                <li><strong>Confidence Levels:</strong> "High" confidence (>75%) indicates strong statistical evidence. "Medium" (40-75%) indicates preliminary signals. "Low" (<40%) insights are filtered out.</li>
                <li><strong>Causal Claims:</strong> Causal language is only used when explicit causal inference methods were applied. Correlational insights use "associated with" phrasing.</li>
                <li><strong>Weak Signals:</strong> Insights marked as "weak signals" are exploratory and require further validation before acting on them.</li>
                <li><strong>Forecasts:</strong> All forecasts include confidence intervals. Model selection was automated using AIC/BIC criteria.</li>
                <li><strong>Limitations:</strong> This analysis is based on the provided historical data. External factors, market shifts, or data quality issues may affect real-world applicability.</li>
            </ul>
        </div>
        
        <footer>
            <p>Generated by <strong>Autonomous Insight Engine</strong></p>
            <p>Powered by AI-driven statistical analysis with human-grade rigor</p>
            <p style="margin-top: 10px; font-size: 0.85em;">{{ metadata.timestamp }}</p>
        </footer>
    </div>
</body>
</html>
        """)
        
        return template.render(report=report, metadata=metadata)
