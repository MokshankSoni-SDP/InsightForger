"""
Self-healing execution loop.

Executes Python code with automatic error correction via LLM.
Enhanced with process isolation, Windows-compatible timeout protection, 
sandboxed namespace, and healing memory.
"""
import os
import time
import traceback
import multiprocessing
import ast
import json
import difflib
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
from litellm import completion
import polars as pl
import numpy as np
import pandas as pd # often needed for compatibility
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from utils.schemas import Hypothesis, ComputationResult
from utils.helpers import get_logger

load_dotenv()
logger = get_logger(__name__)

# Fix for multiprocessing on Windows
if os.name == 'nt':
    # Ensure this is called only once
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass


def _execute_in_process(code: str, df_parquet_path: str, result_queue: multiprocessing.Queue):
    """
    Worker function to run in a separate process.
    Reads DataFrame from parquet to avoid pickling large objects directly
    if we wanted to optimize, but for now we'll pass df via argument if small,
    or reloading is safer for isolation.
    
    To keep it simple and robust:
    We will accept the DF as a path (loading it fresh is safer for isolation).
    """
    try:
        # Re-load data for complete isolation
        df = pl.read_parquet(df_parquet_path)
        
        # Sandbox setup
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
            'enumerate': enumerate, 'float': float, 'int': int, 'len': len,
            'list': list, 'max': max, 'min': min, 'range': range, 'round': round,
            'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
            'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
            'Exception': Exception, 'print': print,
            '__import__': __import__
        }
        
        namespace = {
            '__builtins__': safe_builtins,
            'df': df,
            'pl': pl,
            'np': np,
            'pd': pd,
            'stats': stats,
            'pearsonr': pearsonr,
            'spearmanr': spearmanr,
            'StandardScaler': StandardScaler,
            'LinearRegression': LinearRegression,
            'result': {}
        }
        
        # Execute
        exec(code, namespace)
        
        # Extract result
        result = namespace.get('result', {})
        
        # Validate result type (must be pickle-able and dict)
        if not isinstance(result, dict):
             result_queue.put({"success": False, "error": "Code must set 'result' as a dictionary"})
             return

        # Ensure result is serializable
        # We might need to convert some polars/numpy types to python native
        # but let's try standard pickle first.
        result_queue.put({"success": True, "data": result})
        
    except Exception as e:
        result_queue.put({"success": False, "error": str(e), "trace": traceback.format_exc(), "type": type(e).__name__})


class SelfHealingExecutor:
    """Executes code with self-healing retry mechanism and security."""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.api_key = os.getenv("GROQ_API_KEY_4")
        self.model = os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile")
        
        # Ensure model has groq/ prefix for litellm if it's a groq model
        if "llama" in self.model and not self.model.startswith("groq/"):
             self.model = f"groq/{self.model}"
             
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.execution_timeout = int(os.getenv("EXECUTION_TIMEOUT", "30"))  # seconds
        
        self.healing_history: Dict[str, List[Dict[str, str]]] = {}
        
        # Temporary storage for process isolation
        self.temp_dir = "temp_execution"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.df_path = os.path.join(self.temp_dir, "current_data.parquet")
        
        # Save DF once for workers to read
        # In a real high-throughput system this might be slow, but for this agent it's safe
        self.df.write_parquet(self.df_path)

    def execute_hypothesis(self, hypothesis: Hypothesis) -> ComputationResult:
        """
        Execute hypothesis computation with self-healing and process isolation.
        """
        logger.info(f"Executing hypothesis: {hypothesis.id}")
        
        code = hypothesis.computation_plan
        retry_count = 0
        start_time = time.time()
        
        if hypothesis.id not in self.healing_history:
            self.healing_history[hypothesis.id] = []
            
        attempted_codes = set()
        
        while retry_count <= self.max_retries:
            try:
                # static analysis guardrail
                if not self._validate_syntax_safety(code):
                    raise ValueError("Code violates safety policies (uses forbidden modules or operations)")
                
                # Execute in separate process
                result_data = self._run_in_process(code)
                
                execution_time = time.time() - start_time
                logger.info(f"✓ Hypothesis {hypothesis.id} executed successfully (retries: {retry_count})")
                
                return ComputationResult(
                    hypothesis_id=hypothesis.id,
                    success=True,
                    result_data=result_data,
                    execution_time=execution_time,
                    retry_count=retry_count
                )
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                error_type = type(e).__name__
                
                # If we got a detailed error dict from the worker, use it
                full_trace = error_msg
                if hasattr(e, 'trace'):
                    full_trace = e.trace
                
                logger.warning(f"Execution failed (attempt {retry_count}/{self.max_retries + 1}): {error_type}: {error_msg}")
                
                if retry_count > self.max_retries:
                    return ComputationResult(
                        hypothesis_id=hypothesis.id,
                        success=False,
                        error_message=error_msg,
                        result_data=None,
                        execution_time=time.time() - start_time,
                        retry_count=retry_count - 1
                    )
                
                # 1. Try simple regex fix
                simple_fix = self._try_simple_fix(code, error_type, error_msg, hypothesis)
                if simple_fix:
                    logger.info("✓ Applied semantic guardrail fix")
                    code = simple_fix
                    continue
                
                # 2. Try LLM healing
                logger.info(f"Attempting self-healing for {hypothesis.id}...")
                healed_code = self._heal_code(code, error_msg, full_trace, error_type, hypothesis)
                
                if healed_code and healed_code != code and healed_code not in attempted_codes:
                    code = healed_code
                    attempted_codes.add(code)
                    logger.info("Code healed, retrying execution")
                else:
                    logger.error("Healing failed or produced duplicate/empty code.")
                    # If healing fails, we abort this cycle
                    return ComputationResult(
                        hypothesis_id=hypothesis.id,
                        success=False,
                        error_message=f"Healing failed: {error_msg}",
                        result_data=None,
                        execution_time=time.time() - start_time,
                        retry_count=retry_count
                    )

        # Should not reach here
        return ComputationResult(hypothesis_id=hypothesis.id, success=False, error_message="Unknown error", result_data=None, retry_count=retry_count)

    @staticmethod
    def _validate_syntax_safety(code: str) -> bool:
        """Execute basic AST safety checks on generated code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            if name.name in ['os', 'sys', 'subprocess', 'shutil', 'socket']:
                                logger.error(f"Safety Violation: Forbidden import '{name.name}'")
                                return False
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module
                        if module in ['os', 'sys', 'subprocess', 'shutil', 'socket']:
                            logger.error(f"Safety Violation: Forbidden import from '{module}'")
                            return False
                
                # Check for open() calls - rough check
                if isinstance(node, ast.Call):
                     if isinstance(node.func, ast.Name) and node.func.id == 'open':
                         logger.error("Safety Violation: Use of 'open()' is forbidden")
                         return False
            return True
        except SyntaxError as e:
            logger.error(f"Safety Violation: Syntax Error in generated code: {e}")
            return False # Let execution fail naturally if syntax is bad, or treat as unsafe
    
    def _run_in_process(self, code: str) -> Dict[str, Any]:
        """
        Run code in a separate process with hard timeout.
        This works on Windows because we terminate the process handle.
        """
        
        # We need a way to get the result back. Queue is thread/process safe.
        queue = multiprocessing.Queue()
        
        # Create the process
        # We target the module-level function `_execute_in_process` so it's picklable
        p = multiprocessing.Process(
            target=_execute_in_process,
            args=(code, self.df_path, queue)
        )
        
        p.start()
        
        # Wait for completion or timeout
        p.join(timeout=self.execution_timeout)
        
        if p.is_alive():
            logger.error(f"⏱️ Execution timed out after {self.execution_timeout}s - Killing process")
            p.terminate()
            p.join() # Clean up resources
            raise TimeoutError(f"Execution timed out after {self.execution_timeout} seconds")
            
        # If process finished, check the queue
        if queue.empty():
            # Process died without writing to queue (Segfault, MemoryError, or harsh kill)
            if p.exitcode != 0:
                raise RuntimeError(f"Process crashed with exit code {p.exitcode}")
            else:
                raise RuntimeError("Process finished but returned no result")
            
        result_packet = queue.get()
        
        if not result_packet["success"]:
            # Reconstruct the exception to bubble up
            error_msg = result_packet.get("error", "Unknown Process Error")
            # We attach the trace to the exception object dynamically for the healer
            exc = RuntimeError(error_msg)
            exc.trace = result_packet.get("trace", "")
            raise exc
            
        return result_packet["data"]

    def _try_simple_fix(self, code: str, error_type: str, error_msg: str, hypothesis: Hypothesis) -> Optional[str]:
        """Apply deterministic fixes for common errors."""
        
        # KeyError / Column missing
        if error_type == "KeyError" or "ColumnNotFoundError" in error_type:
             # Extract column name from error message if possible
             # Typical msg: "KeyError: 'foo'" or "Column 'foo' not found"
             import re
             match = re.search(r"'([^']*)'", error_msg)
             if match:
                 missing_col = match.group(1)
                 # find closest match
                 cols = self.df.columns
                 matches = difflib.get_close_matches(missing_col, cols, n=1, cutoff=0.7)
                 if matches:
                     suggestion = matches[0]
                     logger.info(f"Guardrail: Replacing '{missing_col}' with '{suggestion}'")
                     return code.replace(f"'{missing_col}'", f"'{suggestion}'").replace(f'"{missing_col}"', f'"{suggestion}"')
        
        # Division by zero
        if "division by zero" in error_msg.lower() or error_type == "ZeroDivisionError":
             if "epsilon" not in code:
                 return "epsilon = 1e-9\n" + code.replace("/", "/ (").replace(";", ") + epsilon;") # flawed heuristic but simple try
        
        return None

    def _heal_code(self, code: str, error_msg: str, error_trace: str, error_type: str, hypothesis: Hypothesis) -> Optional[str]:
        """Ask LLM to fix the code."""
        
        # Context history
        history = self.healing_history.get(hypothesis.id, [])
        history_txt = "\n".join([f"- Prevented Error: {h['error']}" for h in history[-2:]])
        
        prompt = f"""
You are an expert Python Data Science debugger. Fix the code execution error.

CONTEXT:
Hypothesis: {hypothesis.title}
Metric: {hypothesis.resolved_metric}
Error Type: {error_type}
Error Message: {error_msg}

TRACEBACK:
{error_trace[-600:]}

PREVIOUS ATTEMPTS:
{history_txt}

DF SCHEMA:
Columns: {self.df.columns}

BROKEN CODE:
```python
{code}
```

INSTRUCTIONS:
1. Return ONLY the full, corrected Python code inside ```python``` blocks.
2. Fix the specific error shown.
3. Ensure 'result' dictionary is assigned at the end.
4. Do NOT use markdown outside the code blocks.
5. Do NOT change the logic intent, just fix the bug (e.g., column names, syntax, types).
"""
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                temperature=0.1
            )
            content = response.choices[0].message.content
            
            # Extract code block
            if "```python" in content:
                return content.split("```python")[1].split("```")[0].strip()
            if "```" in content:
                # fallback if they forgot python tag
                return content.split("```")[1].split("```")[0].strip()
            
            # Fallback: assume whole message is code if no blocks
            return content.strip()
            
        except Exception as e:
            logger.error(f"LLM Healing failed: {e}")
            return None
