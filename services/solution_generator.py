import json
from enum import Enum
from typing import List, Any, Optional

import streamlit as st
from openai import OpenAI
from google import genai
from google.genai import types

from .models import ClusterModel, ClusterSolution


class GeminiModels(Enum):
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"


class OpenAIModels(Enum):
    GPT_4_1_MINI = "gpt-4.1-mini"


ModelType = GeminiModels | OpenAIModels


class SolutionGenerator:
    """
    Turns clustered negative feedback into problem summaries + actionable solutions + metrics.
    Uses Chat Completions for maximum compatibility (avoids Responses API input typing issues).
    Requires OPENAI_API_KEY & GEMINI_API_KEY unless a compatible client is passed.
    """

    def __init__(self, model: ModelType = GeminiModels.GEMINI_2_5_FLASH):
        self.model = model
        self.client = self.get_llm_client(model)

    def available(self) -> bool:
        return self.client is not None

    def get_llm_client(self, model: ModelType):
        if isinstance(model, GeminiModels):
            print(f"Initializing Google Client for {model.value}...")
            return genai.Client(api_key=st.secrets["api"]["gemini"])

        elif isinstance(model, OpenAIModels):
            print(f"Initializing OpenAI Client for {model.value}...")
            return OpenAI(api_key=st.secrets["api"]["openai"])

        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    def generate_for_clusters(
        self,
        clusters: List[ClusterModel],
        app_name: str,
        extra_instructions: str = "",
    ) -> List[ClusterSolution]:
        if not self.available():
            raise RuntimeError(
                "OpenAI/Gemini client not available. Set Streamlit secrets for both APIs."
            )

        payload = [
            {
                "cluster_id": c.id,
                "size": c.size,
                "keywords": c.keywords,
                "examples": c.examples,
            }
            for c in clusters
        ]

        system_prompt = (
            "You are a product triage assistant. Users left negative reviews for a mobile app.\n"
            "Given clusters (keywords + representative examples), return a JSON object with key 'clusters': "
            "a list of items {cluster_id, problem_summary (1–2 sentences), key_signals (3–6), solutions (3–6), metrics (2–4)}.\n"
            "Be concrete and engineering-oriented. Avoid vague language."
        )
        if extra_instructions:
            system_prompt += "\nExtra constraints: " + extra_instructions.strip()

        user_prompt = (
            f"App: {app_name}\n"
            f"Problem clusters (JSON):\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            "Return only valid JSON with the schema described above."
        )

        txt: str

        if isinstance(self.model, GeminiModels) and isinstance(
            self.client, genai.Client
        ):  # GeminiModels
            contents = types.Content(
                role="user", parts=[types.Part.from_text(text=user_prompt)]
            )

            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1, include_thoughts=False
                ),
                temperature=0.1,
                tools=None,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
                system_instruction=[types.Part.from_text(text=system_prompt)],
                response_mime_type="application/json",
            )

            txt = (
                self.client.models.generate_content(
                    model=self.model.value,
                    contents=contents,
                    config=config,
                )._get_text()
                or ""
            )
        elif isinstance(self.model, OpenAIModels) and isinstance(
            self.client, OpenAI
        ):  # OpenAIModels
            chat = self.client.chat.completions.create(
                model=self.model.value,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            txt = (chat.choices[0].message.content or "").strip()

        out: List[ClusterSolution] = []
        try:
            data = json.loads(txt)
            for it in data.get("clusters", []):
                out.append(
                    ClusterSolution(
                        cluster_id=int(it.get("cluster_id", -1)),
                        problem_summary=str(it.get("problem_summary", "")).strip(),
                        key_signals=[str(x) for x in it.get("key_signals", [])],
                        solutions=[str(x) for x in it.get("solutions", [])],
                        metrics=[str(x) for x in it.get("metrics", [])],
                    )
                )
        except Exception:
            # If the model didn't return clean JSON, return the raw text as a single solution entry.
            out.append(
                ClusterSolution(
                    cluster_id=-1,
                    problem_summary="LLM returned unstructured text.",
                    key_signals=[],
                    solutions=[txt],
                    metrics=[],
                )
            )
        return out
