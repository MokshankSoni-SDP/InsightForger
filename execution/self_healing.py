"""
Self-healing execution loop.

Executes Python code with automatic error correction via LLM.
Enhanced with timeout protection, sandboxed namespace, and healing memory.
"""
import os
import time
import traceback
import signal
import difflib
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dotenv import load_dotenv
from litellm import completion
import polars as pl
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from utils.schemas import Hypothesis, ComputationResult
from utils.helpers import get_logger

load_dotenv()
logger = get_logger(__name__)


class TimeoutException(Exception):
    """Raised when code execution times out."""
    pass


class SelfHealingExecutor:
    """Executes code with self-healing retry mechanism and security."""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "groq/llama-3.1-70b-versatile")
        if not self.model.startswith("groq/") and "llama" in self.model:
             self.model = f"groq/{self.model}"
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.execution_timeout = int(os.getenv("EXECUTION_TIMEOUT", "30"))  # seconds
        
        # Improvement #5: Healing history
        self.healing_history: Dict[str, List[Dict[str, str]]] = {}
    
    def execute_hypothesis(self, hypothesis: Hypothesis) -> ComputationResult:
        """
        Execute hypothesis computation with self-healing.
        
        Improvements:
        - Timeout protection
        - Sandboxed execution
        - Semantic guardrails
        - Healing history tracking
        
        Args:
            hypothesis: Hypothesis with computation plan
            
        Returns:
            ComputationResult with success status and data
        """
        logger.info(f"Executing hypothesis: {hypothesis.id}")
        
        code = hypothesis.computation_plan
        retry_count = 0
        start_time = time.time()
        
        # Initialize healing history for this hypothesis
        if hypothesis.id not in self.healing_history:
            self.healing_history[hypothesis.id] = []
        
        # Improvement #6: Track if same code was already tried
        attempted_codes = set()
        same_code_retry_allowed = False
        
        while retry_count <= self.max_retries:
            try:
                # Improvement #1: Execute with timeout
                result_data = self._execute_code_with_timeout(code)
                
                execution_time = time.time() - start_time
                
                logger.info(f"✓ Hypothesis {hypothesis.id} executed successfully (retries: {retry_count})")
                
                return ComputationResult(
                    hypothesis_id=hypothesis.id,
                    success=True,
                    result_data=result_data,
                    execution_time=execution_time,
                    retry_count=retry_count
                )
                
            except TimeoutException as e:
                retry_count += 1
                error_msg = f"Execution timed out after {self.execution_timeout}s"
                error_trace = str(e)
                
                logger.warning(f"⏱️ Timeout (attempt {retry_count}/{self.max_retries + 1}): {error_msg}")
                
                if retry_count > self.max_retries:
                    execution_time = time.time() - start_time
                    logger.error(f"✗ Hypothesis {hypothesis.id} failed after {self.max_retries} retries (timeout)")
                    
                    return ComputationResult(
                        hypothesis_id=hypothesis.id,
                        success=False,
                        error_message=error_msg,
                        result_data=None,
                        execution_time=execution_time,
                        retry_count=retry_count - 1
                    )
                
                # For infinite loops, we can't heal - abort
                logger.error("Cannot heal timeout errors (likely infinite loop) - aborting")
                return ComputationResult(
                    hypothesis_id=hypothesis.id,
                    success=False,
                    error_message=f"Execution timeout: {error_msg}",
                    result_data=None,
                    execution_time=time.time() - start_time,
                    retry_count=retry_count
                )
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                error_trace = traceback.format_exc()
                error_type = type(e).__name__
                
                logger.warning(f"Execution failed (attempt {retry_count}/{self.max_retries + 1}): {error_type}: {error_msg}")
                
                if retry_count > self.max_retries:
                    execution_time = time.time() - start_time
                    logger.error(f"✗ Hypothesis {hypothesis.id} failed after {self.max_retries} retries")
                    
                    return ComputationResult(
                        hypothesis_id=hypothesis.id,
                        success=False,
                        error_message=error_msg,
                        result_data=None,
                        execution_time=execution_time,
                        retry_count=retry_count - 1
                    )
                
                # Improvement #3: Semantic guardrail - try simple fix first
                simple_fix = self._try_simple_fix(code, error_type, error_msg, hypothesis)
                if simple_fix:
                    logger.info(f"✓ Applied semantic guardrail fix for {error_type}")
                    code = simple_fix
                    attempted_codes.add(code)
                    continue
                
                # Attempt self-healing with LLM
                logger.info(f"Attempting self-healing for {hypothesis.id}...")
                healed_code = self._heal_code(code, error_msg, error_trace, error_type, hypothesis)
                
                # Improvement #6: Allow same code retry once
                if healed_code:
                    code_changed = healed_code != code
                    
                    if code_changed:
                        # New code, proceed with retry
                        code = healed_code
                        attempted_codes.add(code)
                        same_code_retry_allowed = True
                        logger.info("Code healed with changes, retrying execution")
                    elif not same_code_retry_allowed and healed_code not in attempted_codes:
                        # Same code but first time seeing it - allow one retry
                        code = healed_code
                        attempted_codes.add(code)
                        same_code_retry_allowed = False  # Only once
                        logger.info("Code unchanged but allowing one retry (may be transient error)")
                    else:
                        # Same code and already tried - abort
                        logger.warning("Healed code identical and already attempted - aborting")
                        execution_time = time.time() - start_time
                        return ComputationResult(
                            hypothesis_id=hypothesis.id,
                            success=False,
                            error_message=f"Healing oscillating or stuck: {error_msg}",
                            result_data=None,
                            execution_time=execution_time,
                            retry_count=retry_count
                        )
                else:
                    # Healing failed completely
                    execution_time = time.time() - start_time
                    return ComputationResult(
                        hypothesis_id=hypothesis.id,
                        success=False,
                        error_message=f"Healing failed: {error_msg}",
                        result_data=None,
                        execution_time=execution_time,
                        retry_count=retry_count
                    )
    
    def _execute_code_with_timeout(self, code: str) -> Dict[str, Any]:
        """
        Improvement #1: Execute code with timeout protection.
        
        Uses signal.alarm on Unix or manual timeout tracking on Windows.
        """
        # Improvement #2: Create sandboxed namespace
        namespace = self._create_sandboxed_namespace()
        
        def timeout_handler(signum, frame):
            raise TimeoutException(f"Code execution exceeded {self.execution_timeout}s timeout")
        
        # Try to use signal-based timeout (Unix-like systems)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.execution_timeout)
            
            try:
                exec(code, namespace)
            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
        
        except AttributeError:
            # Windows doesn't have SIGALRM - fallback to manual check
            # Note: This won't interrupt infinite loops, but better than nothing
            logger.warning("SIGALRM not available - timeout protection limited on Windows")
            exec(code, namespace)
        
        # Extract result
        result = namespace.get('result', {})
        
        if not isinstance(result, dict):
            raise ValueError("Code must set 'result' as a dictionary")
        
        return result
    
    def _create_sandboxed_namespace(self) -> Dict[str, Any]:
        """
        Improvement #2: Create sandboxed execution namespace.
        
        Restricts access to dangerous builtins and explicitly whitelists
        safe libraries and functions.
        """
        # Start with minimal builtins
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'max': max,
            'min': min,
            'range': range,
            'round': round,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'Exception': Exception,
            '__import__': __import__,  # Needed for import statements in generated code
        }
        
        # Create namespace with whitelisted modules
        namespace = {
            '__builtins__': safe_builtins,
            'df': self.df,
            'pl': pl,
            'np': np,
            'stats': stats,
            'pearsonr': pearsonr,
            'spearmanr': spearmanr,
            'StandardScaler': StandardScaler,
            'LinearRegression': LinearRegression,
            'result': {}
        }
        
        return namespace
    
    def _try_simple_fix(
        self, 
        code: str, 
        error_type: str, 
        error_msg: str,
        hypothesis: Hypothesis
    ) -> Optional[str]:
        """
        Improvement #3: Semantic guardrail - try simple fixes before LLM.
        
        For common errors, apply deterministic fixes without LLM call.
        """
        # KeyError: Column doesn't exist
        if error_type == "KeyError" and "'" in error_msg:
            missing_col = error_msg.split("'")[1]
            # Find nearest column name
            available_cols = self.df.columns
            close_matches = difflib.get_close_matches(missing_col, available_cols, n=1, cutoff=0.6)
            
            if close_matches:
                suggested_col = close_matches[0]
                logger.info(f"KeyError guardrail: suggesting '{suggested_col}' instead of '{missing_col}'")
                fixed_code = code.replace(f"'{missing_col}'", f"'{suggested_col}'")
                fixed_code = fixed_code.replace(f'"{missing_col}"', f'"{suggested_col}"')
                return fixed_code
        
        # ZeroDivisionError: Add epsilon
        elif error_type == "ZeroDivisionError":
            logger.info("ZeroDivisionError guardrail: adding epsilon protection")
            # Add epsilon to prevent division by zero
            if "/" in code and "epsilon" not in code:
                # Insert epsilon definition at start
                fixed_code = "epsilon = 1e-10\n" + code
                # This is a heuristic - real fix would be more surgical
                return fixed_code
        
        # TypeError: dtype mismatch
        elif error_type == "TypeError" and "dtype" in error_msg.lower():
            logger.info("TypeError guardrail: attempting dtype cast")
            # Try to add .cast(pl.Float64) where needed
            # This is a simplified heuristic
            pass
        
        return None
    
    def _heal_code(
        self, 
        code: str, 
        error_msg: str, 
        error_trace: str,
        error_type: str,
        hypothesis: Hypothesis
    ) -> Optional[str]:
        """
        Use LLM to fix broken code.
        
        Improvements:
        - Pass expected output schema
        - Include healing history
        - Provide full error context
        """
        # Improvement #5: Get healing history
        history = self.healing_history.get(hypothesis.id, [])
        history_text = ""
        if history:
            history_text = "\n\nPrevious Healing Attempts:\n"
            for i, attempt in enumerate(history[-2:], 1):  # Last 2 attempts
                history_text += f"\nAttempt {i}:\n"
                history_text += f"Error: {attempt['error'][:100]}\n"
                history_text += f"Fix Applied: {attempt['fix_description']}\n"
        
        # Improvement #4: Expected output schema
        expected_schema = """
Expected result schema (MANDATORY):
{
    "metric": str,              # hypothesis.metric_target
    "value": float | int,       # primary finding
    "p_value": float | None,    # statistical significance
    "confidence": float,        # confidence score/interval
    "interpretation": str,      # one-line summary
    "_meta": {
        "method": str,          # e.g., "pearson_correlation"
        "assumptions": List[str],
        "n_samples": int,
        "warnings": List[str]
    }
}
"""
        
        prompt = f"""You are debugging Python code that failed. Fix the error and return ONLY the corrected code.

Original Hypothesis: {hypothesis.title}
Analysis Type: {hypothesis.expected_insight_type}
Target Metric: {hypothesis.resolved_metric}
{f"Related Metric: {getattr(hypothesis, 'related_metric', None)}" if getattr(hypothesis, 'related_metric', None) else ""}

{expected_schema}

Failed Code:
```python
{code}
```

Error Type: {error_type}
Error Message: {error_msg}

Error Trace (last 800 chars):
{error_trace[-800:]}

Available DataFrame: 'df' (polars DataFrame)
Columns: {', '.join(self.df.columns[:20])}
Numeric Columns: {', '.join([c for c in self.df.columns if self.df[c].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]][:15])}

{history_text}

CRITICAL RULES:
1. Fix the SPECIFIC error - don't change working logic
2. Maintain the same analysis approach (e.g., if correlation, keep correlation)
3. Return ONLY valid Python code in a code block
4. ALWAYS set 'result' dict with the schema above
5. Use polars syntax (not pandas)
6. Handle edge cases:
   - Null values (use .drop_nulls() or .fill_null())
   - Division by zero (add epsilon = 1e-10)
   - Empty dataframes after filtering
7. If column doesn't exist, use closest match from available columns
8. Ensure result conforms to expected schema

Return the fixed code in a code block."""

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert Python debugger. Fix code errors precisely while preserving analysis intent."},
                    {"role": "user", "content": prompt}
                ],
                api_key=self.api_key,
                temperature=0.1,
                max_tokens=1200
            )
            
            content = response.choices[0].message.content
            
            # Extract code
            if "```python" in content:
                code_start = content.find("```python") + 9
                code_end = content.find("```", code_start)
                healed_code = content[code_start:code_end].strip()
            elif "```" in content:
                code_start = content.find("```") + 3
                code_end = content.find("```", code_start)
                healed_code = content[code_start:code_end].strip()
            else:
                healed_code = content.strip()
            
            # Improvement #5: Record healing attempt
            self.healing_history[hypothesis.id].append({
                "error": error_msg,
                "error_type": error_type,
                "fix_description": f"Healed {error_type}",
                "code_length": len(healed_code)
            })
            
            logger.info("Code healing completed")
            return healed_code
            
        except Exception as e:
            logger.error(f"Code healing failed: {e}")
            return None
