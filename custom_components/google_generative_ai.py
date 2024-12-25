import os
import google.generativeai as genai

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.inputs import DropdownInput, FloatInput, IntInput, SecretStrInput
from langflow.inputs.inputs import HandleInput


class GoogleGenerativeAIComponent(LCModelComponent):
    display_name = "Google Generative AI"
    description = "Generate text using Google Generative AI."
    icon = "GoogleGenerativeAI"
    name = "GoogleGenerativeAIModel"

    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_output_tokens",
            display_name="Max Output Tokens",
            info="The maximum number of tokens to generate.",
            value=8192,
        ),
        DropdownInput(
            name="model",
            display_name="Model",
            info="The name of the model to use.",
            options=[
                "gemini-exp-1206",
                "gemini-1.5-pro-latest",
                "gemini-1.0-pro-latest",
                "gemini-1.0-pro-001",
                "gemini-pro"
            ],
            value="gemini-pro",
        ),
        SecretStrInput(
            name="gemini_api_key",
            display_name="Gemini API Key",
            info="The Gemini API Key to use for the Google Generative AI.",
            password=True,
            value="",
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            info="The maximum cumulative probability of tokens to consider when sampling.",
            advanced=True,
            value=0.95,
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=1,
        ),
        IntInput(
            name="top_k",
            display_name="Top K",
            info="Decode using top-k sampling: consider the set of top_k most probable tokens. Must be positive.",
            advanced=True,
            value=64,
        ),
        HandleInput(
            name="output_parser",
            display_name="Output Parser",
            info="The parser to use to parse the output of the model",
            advanced=True,
            input_types=["OutputParser"],
        ),
    ]

    def build_model(self) -> LanguageModel:
        gemini_api_key = self.gemini_api_key
        model_name = self.model
        max_output_tokens = self.max_output_tokens
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p

        if not gemini_api_key:
            raise ValueError("Gemini API Key is required.")

        genai.configure(api_key=gemini_api_key)

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }

        model_instance = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

        class ChatGoogleGenerativeAI(LanguageModel):
            def __init__(self, model, **kwargs):
                self.model = model
                self.kwargs = kwargs

            def __call__(self, prompt, stop=None, **kwargs):
                chat = self.model.start_chat()
                response = chat.send_message(prompt, stream=False)
                return response.text

        return ChatGoogleGenerativeAI(
            model=model_instance,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
