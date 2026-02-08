"""
LLM Client with Automatic Fallback.

Provides a unified interface for LLM calls with automatic fallback to backup models
when rate limits or errors are encountered.
"""
import os
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from groq import Groq
import requests
from utils.helpers import get_logger

load_dotenv()
logger = get_logger(__name__)


class ModelConfig:
    """Configuration for a single model."""
    def __init__(self, name: str, provider: str, api_key: str):
        self.name = name
        self.provider = provider  # "groq" or "huggingface"
        self.api_key = api_key
        self.client = None
        self.failed = False
        self.last_error = None
    
    def __str__(self):
        return f"{self.provider}/{self.name}"


class FallbackLLMClient:
    """
    LLM client with automatic fallback support.
    
    Tries models in order:
    1. Primary (Groq llama-3.3-70b-versatile)
    2. Fallback 1 (HuggingFace Meta-Llama-3-70B)
    3. Fallback 2 (HuggingFace Mixtral-8x7B)
    """
    
    def __init__(self, token_tracker=None):
        self.token_tracker = token_tracker
        self.models = self._load_models()
        self.current_model_index = 0
        self.call_count = 0
        
        logger.info("🔧 Initialized Fallback LLM Client with 4 models")
        logger.info(f"  Primary: {self.models[0]}")
        logger.info(f"  Fallback 1: {self.models[1]}")
        logger.info(f"  Fallback 2: {self.models[2]}")
        logger.info(f"  Fallback 3: {self.models[3]}")
    
    def _load_models(self) -> List[ModelConfig]:
        """Load model configurations from environment."""
        models = []
        
        # Primary model (Groq)
        primary_model = os.getenv("PRIMARY_MODEL", "groq/llama-3.3-70b-versatile")
        primary_key = os.getenv("PRIMARY_API_KEY") or os.getenv("GROQ_API_KEY")
        
        if primary_model.startswith("groq/"):
            model_name = primary_model.replace("groq/", "")
            models.append(ModelConfig(model_name, "groq", primary_key))
        
        # Fallback 1 (Groq - same model, different instance for rate limit retry)
        fallback1_model = os.getenv("FALLBACK_MODEL_1", "groq/llama-3.3-70b-versatile")
        fallback1_key = os.getenv("FALLBACK_API_KEY_1") or os.getenv("GROQ_API_KEY")
        
        if fallback1_model.startswith("groq/"):
            model_name = fallback1_model.replace("groq/", "")
            models.append(ModelConfig(model_name, "groq", fallback1_key))
        
        # Fallback 2 (HuggingFace)
        fallback2_model = os.getenv("FALLBACK_MODEL_2", "huggingface/meta-llama/Meta-Llama-3-70B-Instruct")
        fallback2_key = os.getenv("FALLBACK_API_KEY_2") or os.getenv("HF_API_KEY")
        
        if fallback2_model.startswith("huggingface/"):
            model_name = fallback2_model.replace("huggingface/", "")
            models.append(ModelConfig(model_name, "huggingface", fallback2_key))
        
        # Fallback 3 (HuggingFace)
        fallback3_model = os.getenv("FALLBACK_MODEL_3", "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1")
        fallback3_key = os.getenv("FALLBACK_API_KEY_3") or os.getenv("HF_API_KEY")
        
        if fallback3_model.startswith("huggingface/"):
            model_name = fallback3_model.replace("huggingface/", "")
            models.append(ModelConfig(model_name, "huggingface", fallback3_key))
        
        return models
    
    def _get_current_model(self) -> ModelConfig:
        """Get the current active model."""
        return self.models[self.current_model_index]
    
    def _switch_to_fallback(self):
        """Switch to the next fallback model."""
        if self.current_model_index < len(self.models) - 1:
            old_model = self.models[self.current_model_index]
            old_model.failed = True
            
            self.current_model_index += 1
            new_model = self.models[self.current_model_index]
            
            logger.warning(f"⚠️  Switching from {old_model} to {new_model}")
            logger.warning(f"   Reason: {old_model.last_error}")
            return True
        else:
            logger.error("❌ All models exhausted!")
            return False
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "429",
            "too many requests",
            "quota exceeded",
            "limit exceeded"
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 500,
        phase: str = "unknown",
        purpose: str = "LLM call"
    ) -> Dict[str, Any]:
        """
        Make a chat completion call with automatic fallback.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            phase: Pipeline phase (for tracking)
            purpose: Purpose of the call (for tracking)
            
        Returns:
            Response dict with 'content', 'usage', 'model_used'
        """
        self.call_count += 1
        attempt = 0
        max_attempts = len(self.models)
        
        while attempt < max_attempts:
            model = self._get_current_model()
            attempt += 1
            
            # Log which model is being used
            if attempt == 1:
                logger.info(f"📞 LLM Call #{self.call_count}: Using PRIMARY model ({model})")
            else:
                logger.info(f"📞 LLM Call #{self.call_count}: Using FALLBACK {attempt-1} ({model})")
            
            try:
                if model.provider == "groq":
                    response = self._call_groq(model, messages, temperature, max_tokens)
                else:  # huggingface
                    response = self._call_huggingface(model, messages, temperature, max_tokens)
                
                # Track tokens
                if self.token_tracker:
                    self.token_tracker.record_call(
                        phase=phase,
                        purpose=purpose,
                        model=f"{model.provider}/{model.name}",
                        input_tokens=response['usage']['input_tokens'],
                        output_tokens=response['usage']['output_tokens']
                    )
                
                # Success!
                response['model_used'] = str(model)
                response['fallback_level'] = attempt - 1  # 0 = primary, 1 = fallback1, 2 = fallback2
                return response
                
            except Exception as e:
                model.last_error = str(e)
                
                if self._is_rate_limit_error(e):
                    logger.warning(f"⚠️  Rate limit hit on {model}")
                    if not self._switch_to_fallback():
                        raise Exception("All models exhausted due to rate limits")
                else:
                    # Non-rate-limit error, re-raise
                    logger.error(f"❌ Error with {model}: {e}")
                    raise
        
        raise Exception("Failed to get response from any model")
    
    def _call_groq(
        self,
        model: ModelConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Call Groq API."""
        if not model.client:
            model.client = Groq(api_key=model.api_key)
        
        response = model.client.chat.completions.create(
            model=model.name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            'content': response.choices[0].message.content,
            'usage': {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }
    
    def _call_huggingface(
        self,
        model: ModelConfig,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Call HuggingFace API."""
        api_url = "https://api-inference.huggingface.co/models/" + model.name
        
        headers = {
            "Authorization": f"Bearer {model.api_key}",
            "Content-Type": "application/json"
        }
        
        # Convert messages to prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 429:
            raise Exception("Rate limit exceeded")
        
        response.raise_for_status()
        result = response.json()
        
        # Extract generated text
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
        else:
            generated_text = result.get('generated_text', '')
        
        # Estimate tokens (HF doesn't return token counts)
        import tiktoken
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        input_tokens = len(encoder.encode(prompt))
        output_tokens = len(encoder.encode(generated_text))
        
        return {
            'content': generated_text,
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_calls': self.call_count,
            'current_model': str(self._get_current_model()),
            'fallback_level': self.current_model_index,
            'models_status': [
                {
                    'model': str(m),
                    'failed': m.failed,
                    'last_error': m.last_error
                }
                for m in self.models
            ]
        }
