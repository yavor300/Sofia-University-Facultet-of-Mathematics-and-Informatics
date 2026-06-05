"""Local Ollama HTTP client for the required Llama MVP layer."""

from __future__ import annotations

import json
from typing import Any

import requests


OLLAMA_SETUP_MESSAGE = (
    "Ollama is unavailable. Start the local Llama runtime with:\n"
    "ollama pull llama3.1:8b\n"
    "ollama serve"
)


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate plain text from local Ollama, returning a clear fallback on failure."""
        payload = self._build_payload(prompt=prompt, system_prompt=system_prompt)
        response = self._post_generate(payload)
        if response.get("error"):
            return response["error"]

        return str(response.get("response", "")).strip()

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        """Generate and parse strict JSON from local Ollama."""
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            output_format="json",
        )
        response = self._post_generate(payload)
        if response.get("error"):
            return {"error": response["error"]}

        raw_response = str(response.get("response", "")).strip()
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return {
                "error": "Ollama returned a non-JSON response.",
                "raw_response": raw_response,
            }

        if not isinstance(parsed, dict):
            return {
                "error": "Ollama returned JSON, but not a JSON object.",
                "raw_response": parsed,
            }

        return parsed

    def _post_generate(self, payload: dict[str, Any]) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as exc:
            return {"error": f"{OLLAMA_SETUP_MESSAGE}\n\nDetails: {exc}"}
        except ValueError as exc:
            return {"error": f"Ollama returned invalid JSON from /api/generate. Details: {exc}"}

        if not isinstance(data, dict):
            return {"error": "Ollama returned an unexpected response shape from /api/generate."}
        if "response" not in data:
            return {"error": "Ollama response is missing the 'response' field."}

        return data

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None = None,
        output_format: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": str(prompt),
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if output_format:
            payload["format"] = output_format

        return payload
