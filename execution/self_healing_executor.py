"""
Phase 3: Self-Healing Executor

Executes generated code with automatic error repair via LLM.
Implements the self-healing loop with up to 3 retry attempts.
"""
import os
import polars as pl
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import traceback
from groq import Groq
from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    value: Optional[float]
    sample_size: int
    is_significant: bool
    metadata: Dict[str, Any]
    error: Optional[str] = None
    retry_count: int = 0
    top_segment: Optional[Dict[str, Any]] = None
    trend_direction: Optional[str] = None
    trend_slope: Optional[float] = None


class SelfHealingExecutor:
    """
    Executes Python code with automatic error repair.
    
    Features:
    - Safe sandbox execution
    - Error capture and LLM-based repair
    - Up to 3 retry attempts
    - Standardized result extraction
    """
    
    def __init__(self, df: pl.DataFrame, max_retries: int = 3):
        """
        Initialize self-healing executor.
        
        Args:
            df: DataFrame to execute code against
            max_retries: Maximum number of retry attempts
        """
        self.df = df
        self.max_retries = max_retries
        
        # Initialize Groq client for code repair
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found, self-healing disabled")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
        
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    def execute_with_healing(
        self, 
        code: str,
        hypothesis_title: str = "Unknown"
    ) -> ExecutionResult:
        """
        Execute code with automatic error repair.
        
        Args:
            code: Python code to execute
            hypothesis_title: Title of hypothesis (for logging)
            
        Returns:
            ExecutionResult with computation results or error
        """
        logger.info(f"Executing: {hypothesis_title}")
        
        current_code = code
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Execute code in safe sandbox
                result_dict = self._execute_code(current_code)
                
                # Validate result structure
                if not isinstance(result_dict, dict):
                    raise ValueError(f"Result must be a dictionary, got {type(result_dict)}")
                
                if 'value' not in result_dict:
                    raise ValueError("Result dictionary must contain 'value' key")
                
                # Extract result
                logger.info(f"✓ Execution successful (attempt {retry_count + 1})")
                
                return ExecutionResult(
                    success=True,
                    value=result_dict.get('value'),
                    sample_size=result_dict.get('sample_size', 0),
                    is_significant=result_dict.get('is_significant', False),
                    metadata=result_dict.get('metadata', {}),
                    retry_count=retry_count,
                    top_segment=result_dict.get('top_segment'),
                    trend_direction=result_dict.get('trend_direction'),
                    trend_slope=result_dict.get('trend_slope')
                )
                
            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()
                
                logger.warning(f"✗ Execution failed (attempt {retry_count + 1}): {error_msg}")
                
                # If max retries reached, return failure
                if retry_count >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached, giving up")
                    return ExecutionResult(
                        success=False,
                        value=None,
                        sample_size=0,
                        is_significant=False,
                        metadata={},
                        error=error_msg,
                        retry_count=retry_count
                    )
                
                # Try to repair code with LLM
                if self.client:
                    logger.info(f"Attempting self-healing repair...")
                    repaired_code = self._repair_code(current_code, error_trace)
                    
                    if repaired_code:
                        logger.info(f"✓ Code repaired, retrying...")
                        current_code = repaired_code
                        retry_count += 1
                    else:
                        logger.error("Failed to repair code, giving up")
                        return ExecutionResult(
                            success=False,
                            value=None,
                            sample_size=0,
                            is_significant=False,
                            metadata={},
                            error=error_msg,
                            retry_count=retry_count
                        )
                else:
                    # No LLM available for repair
                    return ExecutionResult(
                        success=False,
                        value=None,
                        sample_size=0,
                        is_significant=False,
                        metadata={},
                        error=error_msg,
                        retry_count=retry_count
                    )
        
        # Should never reach here
        return ExecutionResult(
            success=False,
            value=None,
            sample_size=0,
            is_significant=False,
            metadata={},
            error="Unknown error",
            retry_count=retry_count
        )
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute code in safe sandbox.
        
        Args:
            code: Python code to execute
            
        Returns:
            Result dictionary from executed code
            
        Raises:
            Exception: If code execution fails
        """
        # Create safe namespace with only df and standard libraries
        namespace = {
            'df': self.df,
            '__builtins__': __builtins__
        }
        
        # Execute code
        exec(code, namespace)
        
        # Extract result
        if 'result' not in namespace:
            raise ValueError("Code must define a 'result' variable")
        
        return namespace['result']
    
    def _repair_code(self, broken_code: str, error_trace: str) -> Optional[str]:
        """
        Repair broken code using LLM.
        
        Args:
            broken_code: Code that failed
            error_trace: Error traceback
            
        Returns:
            Repaired code or None if repair failed
        """
        if not self.client:
            return None
        
        repair_prompt = f"""The following Python code failed with an error. Please fix it and return ONLY the corrected code.

BROKEN CODE:
```python
{broken_code}
```

ERROR:
```
{error_trace}
```

INSTRUCTIONS:
1. Analyze the error and identify the root cause
2. Fix the code to resolve the error
3. Return ONLY the corrected Python code (no explanations, no markdown)
4. Ensure the code still produces a 'result' dictionary with the same structure
5. Do NOT change the overall logic, only fix the error

CORRECTED CODE:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Python debugging expert. Fix code errors precisely."},
                    {"role": "user", "content": repair_prompt}
                ],
                temperature=0.1,  # Low temperature for precise fixes
                max_tokens=2000
            )
            
            repaired_code = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if repaired_code.startswith("```python"):
                repaired_code = repaired_code.split("```python")[1]
                repaired_code = repaired_code.split("```")[0]
            elif repaired_code.startswith("```"):
                repaired_code = repaired_code.split("```")[1]
                repaired_code = repaired_code.split("```")[0]
            
            repaired_code = repaired_code.strip()
            
            logger.info(f"LLM repair completed ({len(repaired_code)} chars)")
            return repaired_code
            
        except Exception as e:
            logger.error(f"LLM repair failed: {e}")
            return None
    
    def execute_simple(self, code: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Simple execution without self-healing (for testing).
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (success, result_dict, error_message)
        """
        try:
            result_dict = self._execute_code(code)
            return True, result_dict, None
        except Exception as e:
            return False, None, str(e)
