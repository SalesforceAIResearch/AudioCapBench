#!/usr/bin/env python3
"""
Model inference clients for audio captioning.

Supports:
  - OpenAI Chat Completions (gpt-4o-audio-preview, gpt-audio, etc.)
  - OpenAI Realtime WebSocket (gpt-realtime, gpt-realtime-mini, etc.)
  - Google Gemini (gemini-2.5-flash, gemini-2.5-pro, etc.)
  - Alibaba Qwen-Audio (qwen-audio-turbo via DashScope)

Each client takes an audio file path and a text prompt, and returns the
model's caption as a string.
"""

import base64
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class AudioCaptionModel(ABC):
    """Base class for audio captioning model clients."""

    def __init__(self, model_id: str, max_tokens: int = 256, temperature: float = 0.0):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate_caption(self, audio_path: str, instruction: str) -> Dict[str, Any]:
        """
        Generate a caption for the given audio file.

        Args:
            audio_path: Path to the audio file (WAV format).
            instruction: Text prompt/instruction for the model.

        Returns:
            Dict with keys:
                - output: The generated caption string.
                - inference_time: Time taken in seconds.
                - model_id: The model identifier used.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'gemini')."""
        pass

    def _read_audio_bytes(self, audio_path: str) -> bytes:
        """Read audio file as raw bytes."""
        with open(audio_path, "rb") as f:
            return f.read()

    def _read_audio_base64(self, audio_path: str) -> str:
        """Read audio file and return base64-encoded string."""
        return base64.b64encode(self._read_audio_bytes(audio_path)).decode("utf-8")


# ===================================================================
# OpenAI (GPT-4o audio)
# ===================================================================

class OpenAIAudioModel(AudioCaptionModel):
    """
    OpenAI model with audio input support.

    Authentication (in order of priority):
      1. Standard: OPENAI_API_KEY (works with api.openai.com, default)
      2. Gateway: OPENAI_API_KEY + OPENAI_BASE_URL (for Salesforce Research Gateway)

    When OPENAI_BASE_URL contains 'gateway.salesforceresearch.ai', the key is
    passed via X-Api-Key header with a dummy api_key.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-audio-preview",
        max_tokens: int = 256,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(model_id, max_tokens, temperature)
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        url = base_url or os.environ.get("OPENAI_BASE_URL")

        # Salesforce gateway: pass key via X-Api-Key header
        if url and "gateway.salesforceresearch.ai" in url:
            self.client = OpenAI(
                api_key="dummy",
                base_url=url,
                default_headers={"X-Api-Key": key},
            )
        else:
            # Standard OpenAI API (default)
            self.client = OpenAI(api_key=key, base_url=url if url else None)

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate_caption(self, audio_path: str, instruction: str) -> Dict[str, Any]:
        start = time.time()
        audio_b64 = self._read_audio_base64(audio_path)

        # Determine audio format from extension
        ext = Path(audio_path).suffix.lower().lstrip(".")
        audio_format = ext if ext in ("wav", "mp3", "flac", "ogg") else "wav"

        response = self.client.chat.completions.create(
            model=self.model_id,
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        output = response.choices[0].message.content.strip()
        return {
            "output": output,
            "inference_time": time.time() - start,
            "model_id": self.model_id,
        }


# ===================================================================
# Google Gemini
# ===================================================================

class GeminiAudioModel(AudioCaptionModel):
    """
    Google Gemini model with audio input support.

    Authentication (in order of priority):
      1. Standard: GEMINI_API_KEY or GOOGLE_API_KEY (works with aistudio.google.com, default)
      2. Vertex AI: VERTEX_PROJECT + gcloud auth (for GCP Vertex AI)

    When VERTEX_PROJECT is set or no API key is found, uses Vertex AI backend.
    Otherwise uses the standard Gemini API with the API key.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        max_tokens: int = 256,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        super().__init__(model_id, max_tokens, temperature)
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        vertex_project = project or os.environ.get("VERTEX_PROJECT")

        if vertex_project:
            # Vertex AI mode (GCP authentication via gcloud)
            self.client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=location or os.environ.get("VERTEX_LOCATION", "us-central1"),
            )
        elif key:
            # Standard Gemini API key mode (default for public use)
            self.client = genai.Client(api_key=key)
        else:
            raise ValueError(
                "Gemini requires either GEMINI_API_KEY (standard) or "
                "VERTEX_PROJECT (Vertex AI). Set one in credentials.env."
            )

    @property
    def provider_name(self) -> str:
        return "gemini"

    def generate_caption(self, audio_path: str, instruction: str) -> Dict[str, Any]:
        from google.genai import types

        start = time.time()
        audio_bytes = self._read_audio_bytes(audio_path)

        # Determine MIME type from extension
        ext = Path(audio_path).suffix.lower().lstrip(".")
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mp3",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "aac": "audio/aac",
        }
        mime_type = mime_map.get(ext, "audio/wav")

        # Only add thinking_config for 2.5+ thinking models
        config_kwargs = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if "2.5" in self.model_id or "3" in self.model_id:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=1024,
            )

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                instruction,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Check for truncation
        if hasattr(response, 'candidates') and response.candidates:
            finish = response.candidates[0].finish_reason
            if finish and str(finish) not in ("STOP", "1", "FinishReason.STOP"):
                print(f"    Warning: Gemini finish_reason={finish}")

        output = response.text.strip() if response.text else ""
        # Strip markdown formatting for cleaner evaluation
        import re
        output = re.sub(r'\*\*([^*]+)\*\*', r'\1', output)  # **bold** -> bold
        output = re.sub(r'\*([^*]+)\*', r'\1', output)      # *italic* -> italic
        return {
            "output": output,
            "inference_time": time.time() - start,
            "model_id": self.model_id,
        }


# ===================================================================
# OpenAI Realtime (WebSocket API)
# ===================================================================

class OpenAIRealtimeModel(AudioCaptionModel):
    """
    OpenAI Realtime model with audio input via WebSocket API.

    Uses AsyncOpenAI + client.beta.realtime.connect() for models like
    gpt-realtime, gpt-realtime-mini, gpt-4o-realtime-preview.

    Authentication:
      1. Standard: OPENAI_API_KEY (uses OpenAI's default WebSocket endpoint)
      2. Gateway: SFR_GATEWAY_API_KEY or OPENAI_API_KEY + gateway base URL
         The WebSocket URL uses /openai/ws/v1/ (not /openai/process/v1/).
         If OPENAI_REALTIME_BASE_URL is not set, it is auto-derived from
         OPENAI_BASE_URL by replacing /process/ with /ws/.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-realtime-preview",
        max_tokens: int = 256,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        ws_base_url: Optional[str] = None,
    ):
        super().__init__(model_id, max_tokens, temperature)

        # Resolve WebSocket base URL
        self.ws_base_url = (
            ws_base_url
            or os.environ.get("OPENAI_REALTIME_BASE_URL")
        )
        if not self.ws_base_url:
            # Auto-derive from OPENAI_BASE_URL: /process/ -> /ws/
            rest_url = os.environ.get("OPENAI_BASE_URL", "")
            if "gateway.salesforceresearch.ai" in rest_url:
                self.ws_base_url = rest_url.replace("/process/", "/ws/")

        # Detect if using Salesforce gateway
        self._is_gateway = bool(
            self.ws_base_url and "gateway.salesforceresearch.ai" in self.ws_base_url
        )

        # Gateway auth: prefer SFR_GATEWAY_API_KEY, fall back to OPENAI_API_KEY
        if self._is_gateway:
            self.api_key = (
                api_key
                or os.environ.get("SFR_GATEWAY_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def provider_name(self) -> str:
        return "openai_realtime"

    def generate_caption(self, audio_path: str, instruction: str) -> Dict[str, Any]:
        """Synchronous wrapper around the async realtime captioning."""
        import asyncio

        # Use asyncio.run() if no loop is running, otherwise use a new loop
        try:
            loop = asyncio.get_running_loop()
            # We're inside an existing event loop (e.g., Jupyter) — use a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self._async_generate_caption(audio_path, instruction),
                )
                return future.result()
        except RuntimeError:
            # No event loop running — safe to use asyncio.run()
            return asyncio.run(
                self._async_generate_caption(audio_path, instruction)
            )

    async def _async_generate_caption(
        self, audio_path: str, instruction: str
    ) -> Dict[str, Any]:
        """Send audio to realtime model and collect text response."""
        from openai import AsyncOpenAI

        start = time.time()

        if self._is_gateway:
            client = AsyncOpenAI(
                api_key="dummy",
                base_url=self.ws_base_url,
                default_headers={"X-Api-Key": self.api_key},
            )
            connect_kwargs = {"extra_headers": {"X-Api-Key": self.api_key}}
        else:
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.ws_base_url if self.ws_base_url else None,
            )
            connect_kwargs = {}

        # Read and base64-encode audio
        audio_b64 = self._read_audio_base64(audio_path)

        try:
            async with client.beta.realtime.connect(
                model=self.model_id,
                **connect_kwargs,
            ) as conn:
                # Configure session: text-only output, no voice
                await conn.session.update(
                    session={
                        "modalities": ["text"],
                        "instructions": f"You are an audio captioning assistant. Always respond in English. {instruction}",
                        "temperature": max(0.6, self.temperature),
                    }
                )

                # Send audio as input
                await conn.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "audio": audio_b64,
                            }
                        ],
                    }
                )

                # Request response
                await conn.response.create()

                # Collect text response
                text_parts = []
                async for event in conn:
                    et = getattr(event, "type", None)
                    if et == "response.text.delta":
                        text_parts.append(event.delta)
                    elif et == "response.text.done":
                        break
                    elif et == "response.done":
                        break
                    elif et == "error":
                        error_msg = getattr(event, "error", "")
                        raise RuntimeError(f"Realtime API error: {error_msg}")

                output = "".join(text_parts).strip()

        except TypeError:
            # Fallback: extra_headers not supported in SDK version
            async with client.beta.realtime.connect(
                model=self.model_id,
            ) as conn:
                await conn.session.update(
                    session={
                        "modalities": ["text"],
                        "instructions": f"You are an audio captioning assistant. Always respond in English. {instruction}",
                        "temperature": max(0.6, self.temperature),
                    }
                )
                await conn.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "audio": audio_b64}
                        ],
                    }
                )
                await conn.response.create()

                text_parts = []
                async for event in conn:
                    et = getattr(event, "type", None)
                    if et == "response.text.delta":
                        text_parts.append(event.delta)
                    elif et in ("response.text.done", "response.done"):
                        break
                    elif et == "error":
                        raise RuntimeError(
                            f"Realtime API error: {getattr(event, 'error', '')}"
                        )

                output = "".join(text_parts).strip()

        return {
            "output": output,
            "inference_time": time.time() - start,
            "model_id": self.model_id,
        }


# ===================================================================
# Qwen-Audio (DashScope)
# ===================================================================

class QwenAudioModel(AudioCaptionModel):
    """
    Alibaba Qwen-Audio model via DashScope API.

    Requires DASHSCOPE_API_KEY environment variable.
    Note: 30-second max audio duration.
    """

    def __init__(
        self,
        model_id: str = "qwen-audio-turbo",
        max_tokens: int = 256,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_id, max_tokens, temperature)
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")

    @property
    def provider_name(self) -> str:
        return "qwen_audio"

    def generate_caption(self, audio_path: str, instruction: str) -> Dict[str, Any]:
        import dashscope
        from dashscope import MultiModalConversation

        start = time.time()

        # DashScope accepts local file paths with file:// prefix
        # or base64 with data:;base64, prefix
        audio_b64 = self._read_audio_base64(audio_path)
        audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"audio": audio_data_uri},
                    {"text": instruction},
                ],
            }
        ]

        response = MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model_id,
            messages=messages,
            result_format="message",
        )

        output = ""
        if response and response.output and response.output.choices:
            choice = response.output.choices[0]
            if hasattr(choice, "message") and choice.message:
                content = choice.message.content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            output += item["text"]
                elif isinstance(content, str):
                    output = content

        return {
            "output": output.strip(),
            "inference_time": time.time() - start,
            "model_id": self.model_id,
        }


# ===================================================================
# Factory
# ===================================================================

MODEL_REGISTRY = {
    "openai": OpenAIAudioModel,
    "openai_realtime": OpenAIRealtimeModel,
    "gemini": GeminiAudioModel,
    "qwen_audio": QwenAudioModel,
}


def create_model(
    provider: str,
    model_id: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **kwargs,
) -> AudioCaptionModel:
    """
    Create a model client by provider name.

    Args:
        provider: One of 'openai', 'gemini', 'qwen_audio'.
        model_id: Model identifier (uses provider default if None).
        max_tokens: Max tokens for generation.
        temperature: Sampling temperature.
        **kwargs: Additional provider-specific arguments.

    Returns:
        An AudioCaptionModel instance.
    """
    if provider not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY[provider]
    init_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs,
    }
    if model_id:
        init_kwargs["model_id"] = model_id
    return cls(**init_kwargs)


def list_providers() -> list:
    """Return list of available provider names."""
    return list(MODEL_REGISTRY.keys())
