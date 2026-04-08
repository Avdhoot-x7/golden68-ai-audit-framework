"""
Golden 68 - Model Adapters
Supports multiple LLM providers: Gemini, OpenAI, Anthropic, OpenRouter, NVIDIA
With credit exhaustion handling and API key fallback
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
import google.generativeai as genai


class APIKeyExhaustedError(Exception):
    """Raised when an API key runs out of credits."""
    def __init__(self, provider: str, message: str = ""):
        self.provider = provider
        self.message = message or f"{provider} API key has insufficient credits"
        super().__init__(self.message)


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the model name."""
        pass
    
    @abstractmethod
    def is_credit_error(self, response: str) -> bool:
        """Check if response indicates credit exhaustion."""
        pass


class GeminiAdapter(ModelAdapter):
    """Adapter for Google Gemini models."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name.replace("models/", "")
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Gemini."""
        try:
            generation_config = genai.types.GenerationConfig(temperature=temperature)
            for key in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
                kwargs.pop(key, None)
            
            response = self.model.generate_content(prompt, generation_config=generation_config, **kwargs)
            return response.text
        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ['quota', 'limit', 'rate', 'exhausted', '403', '429']):
                raise APIKeyExhaustedError("Gemini", err)
            return f"Error: {err}"
    
    def get_name(self) -> str:
        return f"Gemini ({self.model_name})"
    
    def is_credit_error(self, response: str) -> bool:
        return False


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.model_name = model_name
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using OpenAI."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ['quota', 'limit', 'rate', 'exhausted', 'insufficient', '429']):
                raise APIKeyExhaustedError("OpenAI", err)
            return f"Error: {err}"
    
    def get_name(self) -> str:
        return f"OpenAI ({self.model_name})"
    
    def is_credit_error(self, response: str) -> bool:
        return False


class OpenRouterAdapter(ModelAdapter):
    """Adapter for OpenRouter models with credit detection."""
    
    def __init__(self, api_key: str, model_name: str = "openai/gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using OpenRouter."""
        try:
            import requests
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 512,
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                data = response.json() if response.text else {}
                error_msg = data.get("error", {}).get("message", response.text)
                
                if response.status_code == 402 or "credits" in error_msg.lower() or "payment" in error_msg.lower():
                    raise APIKeyExhaustedError("OpenRouter", f"HTTP 402 - {error_msg}")
                
                return f"Error: HTTP {response.status_code} - {error_msg}"
            
            data = response.json()
            
            if "error" in data:
                error_msg = data["error"].get("message", str(data["error"]))
                if "credits" in error_msg.lower() or "payment" in error_msg.lower():
                    raise APIKeyExhaustedError("OpenRouter", error_msg)
                return f"Error: {error_msg}"
            
            if "choices" not in data or not data["choices"]:
                return "Error: No choices in response"
            
            return data["choices"][0]["message"]["content"]
            
        except APIKeyExhaustedError:
            raise
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_name(self) -> str:
        return f"OpenRouter ({self.model_name})"
    
    def is_credit_error(self, response: str) -> bool:
        if not response:
            return False
        return "402" in response and "credits" in response.lower()


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        self.api_key = api_key
        self.model_name = model_name
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Anthropic."""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ['quota', 'limit', 'rate', 'exhausted', 'overloaded']):
                raise APIKeyExhaustedError("Anthropic", err)
            return f"Error: {err}"
    
    def get_name(self) -> str:
        return f"Anthropic ({self.model_name})"
    
    def is_credit_error(self, response: str) -> bool:
        return False


class NVIDIAAdapter(ModelAdapter):
    """Adapter for NVIDIA Cloud API (free tier with rate limits)."""
    
    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-120b"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://integrate.api.nvidia.com/v1"
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> str:
        """Generate response using NVIDIA Cloud API with token limit to prevent exhaustion."""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,  # Limit tokens to prevent credit exhaustion
                top_p=1,
                **kwargs
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ['quota', 'limit', 'rate', 'exhausted', 'insufficient', '429', 'rate_limit']):
                raise APIKeyExhaustedError("NVIDIA", err)
            return f"Error: {err}"
    
    def get_name(self) -> str:
        return f"NVIDIA ({self.model_name})"
    
    def is_credit_error(self, response: str) -> bool:
        return False


class ModelAdapterFactory:
    """Factory for creating model adapters."""
    
    @staticmethod
    def create(provider: str, api_key: str, model_name: str) -> ModelAdapter:
        """Create a model adapter based on provider."""
        providers = {
            "gemini": GeminiAdapter,
            "openai": OpenAIAdapter,
            "openrouter": OpenRouterAdapter,
            "anthropic": AnthropicAdapter,
            "nvidia": NVIDIAAdapter,
        }
        
        adapter_class = providers.get(provider.lower())
        if adapter_class is None:
            raise ValueError(f"Unknown provider: {provider}")
        
        return adapter_class(api_key, model_name)
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Return list of available providers."""
        return ["gemini", "openai", "openrouter", "anthropic", "nvidia"]
    
    @staticmethod
    def create_resilient(provider: str, model_name: str, api_keys: List[str], auto_recovery: bool = True) -> 'ResilientModelClient':
        """
        Create a resilient client with optional auto-recovery.
        Auto-recovery automatically switches to fallback models when deprecated.
        """
        if auto_recovery:
            return AutoRecoveryModelClient(provider, model_name, api_keys)
        return ResilientModelClient(provider, model_name, api_keys)


class ResilientModelClient:
    """
    A wrapper that manages API keys with automatic fallback.
    When a key runs out of credits, it will:
    1. Try to use the next fallback key
    2. Raise APIKeyExhaustedError if no keys remain
    """
    
    def __init__(self, provider: str, model_name: str, api_keys: List[str]):
        self.provider = provider
        self.model_name = model_name
        self.api_keys = api_keys
        self.current_key_index = 0
        
        if not api_keys:
            raise ValueError("At least one API key must be provided")
    
    def _get_current_adapter(self) -> ModelAdapter:
        """Get the adapter for the current key."""
        return ModelAdapterFactory.create(
            self.provider,
            self.api_keys[self.current_key_index],
            self.model_name
        )
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate a response, automatically switching keys if one is exhausted.
        Raises APIKeyExhaustedError if all keys are exhausted.
        """
        last_error = None
        
        for i in range(self.current_key_index, len(self.api_keys)):
            try:
                adapter = self._get_current_adapter()
                response = adapter.generate(prompt, temperature, **kwargs)
                
                if adapter.is_credit_error(response):
                    raise APIKeyExhaustedError(
                        self.provider,
                        f"API key #{i+1} returned credit error: {response}"
                    )
                
                return response
                
            except APIKeyExhaustedError as e:
                last_error = e
                self.current_key_index += 1
                
                if self.current_key_index < len(self.api_keys):
                    print(f"⚠️ API key #{i+1} exhausted. Switching to key #{self.current_key_index+1}...")
                else:
                    raise
        
        raise APIKeyExhaustedError(
            self.provider,
            f"All {len(self.api_keys)} API keys are exhausted"
        )


# Fallback models for auto-recovery when models are deprecated
MODEL_FALLBACKS = {
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.0-flash-thinking",
        "gemini-1.5-flash",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ],
    "openrouter": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-opus",
    ],
    "anthropic": [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "nvidia": [
        "openai/gpt-oss-120b",
        "meta/llama-3.1-405b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
    ],
}


class AutoRecoveryModelClient(ResilientModelClient):
    """
    Extended ResilientModelClient with automatic model fallback.
    When a model returns 404 (deprecated), it auto-switches to next available model.
    """
    
    def __init__(self, provider: str, model_name: str, api_keys: List[str]):
        super().__init__(provider, model_name, api_keys)
        self.fallback_tried = []  # Track tried models
    
    def _get_fallback_model(self) -> Optional[str]:
        """Get the next available fallback model."""
        fallbacks = MODEL_FALLBACKS.get(self.provider, [])
        for model in fallbacks:
            if model not in self.fallback_tried and model != self.model_name:
                return model
        return None
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate with auto-recovery from deprecated models."""
        last_error = None
        
        while True:
            try:
                adapter = self._get_current_adapter()
                response = adapter.generate(prompt, temperature, **kwargs)
                
                # Check for deprecation/404 errors
                if self._is_model_deprecated_error(response):
                    self.fallback_tried.append(self.model_name)
                    fallback = self._get_fallback_model()
                    if fallback:
                        print(f"⚠️ Model {self.model_name} deprecated. Switching to {fallback}...")
                        self.model_name = fallback
                        continue
                    else:
                        return f"Error: Model {self.original_model} and all fallbacks deprecated"
                
                if adapter.is_credit_error(response):
                    raise APIKeyExhaustedError(
                        self.provider,
                        f"API key #{self.current_key_index+1} returned credit error"
                    )
                
                return response
                
            except APIKeyExhaustedError:
                raise
    
    def _is_model_deprecated_error(self, response: str) -> bool:
        """Check if response indicates model is deprecated."""
        if not response:
            return False
        response_lower = response.lower()
        return any(x in response_lower for x in [
            "no longer available",
            "not available to new users",
            "404",
            "model not found",
            "deprecated",
            "invalid model"
        ])


def get_status(self) -> Dict[str, Any]:
        """Get status info about the client."""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "total_keys": len(self.api_keys),
            "current_key_index": self.current_key_index,
            "keys_remaining": len(self.api_keys) - self.current_key_index,
        }


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "conf", "config.yaml"
        )
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
