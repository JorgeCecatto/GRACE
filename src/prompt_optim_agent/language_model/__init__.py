from . import *

from .openai_model import OpenAIModel
from .ollama_model import OllamaModel
LANGUAGE_MODELS = {
    "openai":OpenAIModel,
    "ollama": OllamaModel,
}

def get_language_model(language_model_name):
    assert language_model_name in LANGUAGE_MODELS.keys(), f"Language model type {language_model_name} is not supported."
    return LANGUAGE_MODELS[language_model_name]
    
