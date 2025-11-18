from . import *
from .grace_world_model import GraceSearchWorldModel

# Tenta importar módulos opcionais
try:
    from .difficulty_classifier import DifficultyClassifier
except ImportError:
    DifficultyClassifier = None

try:
    from .meta_prompter import MetaPrompter
except ImportError:
    MetaPrompter = None

WORLD_MODELS = {
    'grace': GraceSearchWorldModel,
    }

def get_world_model(world_model_name):
    assert world_model_name in WORLD_MODELS.keys(), f"World model {world_model_name} is not supported."
    return WORLD_MODELS[world_model_name]