"""
Token Tracking Module.

Tracks token usage for all LLM calls with detailed per-call and cumulative statistics.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class LLMCallRecord:
    """Record of a single LLM call."""
    timestamp: str
    phase: str  # e.g., "context_injection", "hypothesis_generation"
    purpose: str  # e.g., "Infer business context", "Generate CFO hypotheses"
    model: str  # e.g., "groq/llama-3.1-70b-versatile"
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: Optional[int] = None
    
    def __str__(self):
        return (
            f"[{self.phase}] {self.purpose}\n"
            f"  Model: {self.model}\n"
            f"  Input: {self.input_tokens:,} tokens\n"
            f"  Output: {self.output_tokens:,} tokens\n"
            f"  Total: {self.total_tokens:,} tokens"
        )


@dataclass
class TokenStats:
    """Cumulative token statistics."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    by_phase: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def add_call(self, record: LLMCallRecord):
        """Add a call record to statistics."""
        self.total_calls += 1
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_tokens += record.total_tokens
        
        # Track by phase
        if record.phase not in self.by_phase:
            self.by_phase[record.phase] = {
                "calls": 0,
                "input": 0,
                "output": 0,
                "total": 0
            }
        self.by_phase[record.phase]["calls"] += 1
        self.by_phase[record.phase]["input"] += record.input_tokens
        self.by_phase[record.phase]["output"] += record.output_tokens
        self.by_phase[record.phase]["total"] += record.total_tokens
        
        # Track by model
        if record.model not in self.by_model:
            self.by_model[record.model] = {
                "calls": 0,
                "input": 0,
                "output": 0,
                "total": 0
            }
        self.by_model[record.model]["calls"] += 1
        self.by_model[record.model]["input"] += record.input_tokens
        self.by_model[record.model]["output"] += record.output_tokens
        self.by_model[record.model]["total"] += record.total_tokens


class TokenTracker:
    """Tracks token usage across all LLM calls in a pipeline run."""
    
    def __init__(self):
        self.calls: List[LLMCallRecord] = []
        self.stats = TokenStats()
        self.start_time = datetime.now()
    
    def record_call(
        self,
        phase: str,
        purpose: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: Optional[int] = None
    ):
        """
        Record an LLM call.
        
        Args:
            phase: Pipeline phase (e.g., "context_injection")
            purpose: Human-readable purpose
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Optional latency in milliseconds
        """
        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            phase=phase,
            purpose=purpose,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms
        )
        
        self.calls.append(record)
        self.stats.add_call(record)
        
        # Log the call
        logger.info(f"📊 LLM Call #{len(self.calls)}: {purpose}")
        logger.info(f"   Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {record.total_tokens:,}")
    
    def get_summary(self) -> str:
        """Get a formatted summary of all token usage."""
        lines = []
        lines.append("=" * 80)
        lines.append("TOKEN USAGE SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall stats
        lines.append(f"Total LLM Calls: {self.stats.total_calls}")
        lines.append(f"Total Input Tokens: {self.stats.total_input_tokens:,}")
        lines.append(f"Total Output Tokens: {self.stats.total_output_tokens:,}")
        lines.append(f"Total Tokens: {self.stats.total_tokens:,}")
        lines.append("")
        
        # Per-call breakdown
        lines.append("-" * 80)
        lines.append("PER-CALL BREAKDOWN")
        lines.append("-" * 80)
        lines.append("")
        
        for i, call in enumerate(self.calls, 1):
            lines.append(f"Call #{i}: {call.purpose}")
            lines.append(f"  Phase: {call.phase}")
            lines.append(f"  Model: {call.model}")
            lines.append(f"  Input Tokens:  {call.input_tokens:>8,}")
            lines.append(f"  Output Tokens: {call.output_tokens:>8,}")
            lines.append(f"  Total Tokens:  {call.total_tokens:>8,}")
            if call.latency_ms:
                lines.append(f"  Latency: {call.latency_ms}ms")
            lines.append("")
        
        # By phase
        lines.append("-" * 80)
        lines.append("BY PHASE")
        lines.append("-" * 80)
        lines.append("")
        
        for phase, stats in sorted(self.stats.by_phase.items()):
            lines.append(f"{phase}:")
            lines.append(f"  Calls: {stats['calls']}")
            lines.append(f"  Input:  {stats['input']:>8,} tokens")
            lines.append(f"  Output: {stats['output']:>8,} tokens")
            lines.append(f"  Total:  {stats['total']:>8,} tokens")
            lines.append("")
        
        # By model
        lines.append("-" * 80)
        lines.append("BY MODEL")
        lines.append("-" * 80)
        lines.append("")
        
        for model, stats in sorted(self.stats.by_model.items()):
            lines.append(f"{model}:")
            lines.append(f"  Calls: {stats['calls']}")
            lines.append(f"  Input:  {stats['input']:>8,} tokens")
            lines.append(f"  Output: {stats['output']:>8,} tokens")
            lines.append(f"  Total:  {stats['total']:>8,} tokens")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_report(self, filepath: str):
        """Save token usage report to file."""
        with open(filepath, 'w') as f:
            f.write(self.get_summary())
        logger.info(f"Token usage report saved to: {filepath}")
    
    def get_cost_estimate(self) -> Dict[str, float]:
        """
        Estimate cost based on token usage.
        
        Returns:
            Dict with cost breakdown by model
        """
        # Pricing (per 1M tokens)
        pricing = {
            "groq/llama-3.1-70b-versatile": {
                "input": 0.59,  # $0.59 per 1M input tokens
                "output": 0.79  # $0.79 per 1M output tokens
            },
            "huggingface/meta-llama/Meta-Llama-3-8B-Instruct": {
                "input": 0.30,
                "output": 0.60
            }
        }
        
        costs = {}
        total_cost = 0.0
        
        for model, stats in self.stats.by_model.items():
            if model in pricing:
                input_cost = (stats["input"] / 1_000_000) * pricing[model]["input"]
                output_cost = (stats["output"] / 1_000_000) * pricing[model]["output"]
                model_cost = input_cost + output_cost
                
                costs[model] = {
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": model_cost
                }
                total_cost += model_cost
        
        costs["total"] = total_cost
        return costs
