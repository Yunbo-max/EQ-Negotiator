"""
LLM wrapper for EQ-Negotiator.
Supports online (GPT, Claude) and offline (DeepSeek-7B, Llama-7B) models.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LLMWrapper:
    """Unified LLM interface for online and offline models."""

    def __init__(self, model_name: str, role: str = "generic"):
        self.model_name = model_name
        self.role = role
        self.model = self._initialize_model(model_name)

    def _initialize_model(self, model_name: str):
        model_lower = model_name.lower()

        # Offline models (SLMs for edge deployment)
        if any(kw in model_lower for kw in ["deepseek", "llama", "offline:"]):
            if not TRANSFORMERS_AVAILABLE:
                print(f"Transformers not available for {self.role}, falling back to GPT-4o-mini")
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

            actual_model = model_name.replace("offline:", "")
            model_map = {
                "deepseek-7b": "deepseek-ai/DeepSeek-LLM-7B-Chat",
                "deepseek": "deepseek-ai/DeepSeek-LLM-7B-Chat",
                "llama-7b": "meta-llama/Llama-2-7b-chat-hf",
                "llama": "meta-llama/Llama-2-7b-chat-hf",
            }
            actual_model = model_map.get(actual_model.lower(), actual_model)

            try:
                return self._init_offline(actual_model)
            except Exception as e:
                print(f"Failed to load {actual_model}: {e}, falling back to GPT-4o-mini")
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        # Online models
        if "gpt" in model_lower:
            temp = 1.0 if "gpt-5" in model_lower else 0.7
            return ChatOpenAI(model=model_name, temperature=temp)
        elif "claude" in model_lower:
            return ChatAnthropic(model=model_name, temperature=0.7)
        else:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def _init_offline(self, model_name: str):
        class OfflineLLM:
            def __init__(self, name):
                self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    name, torch_dtype=dtype, device_map=device, trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            def invoke(self, messages, **kwargs):
                if isinstance(messages, list) and hasattr(messages[0], 'content'):
                    prompt = messages[0].content
                else:
                    prompt = str(messages[0]) if isinstance(messages, list) else str(messages)

                inputs = self.tokenizer(
                    f"User: {prompt}\n\nAssistant:",
                    return_tensors="pt", padding=True, truncation=True, max_length=2048
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=512,
                        temperature=kwargs.get('temperature', 0.7),
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
                ).strip()

                class MockMsg:
                    def __init__(self, content):
                        self.content = content
                return MockMsg(response)

        return OfflineLLM(model_name)

    def invoke(self, messages, **kwargs):
        return self.model.invoke(messages, **kwargs)
