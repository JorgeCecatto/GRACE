"""
Curriculum Learning Module

Pre-processes datasets to classify examples by difficulty
for curriculum learning optimization.
"""

from .curriculum_classifier import (
    CurriculumClassifier,
    CurriculumExample,
    DifficultyLevel
)

__all__ = [
    'CurriculumClassifier',
    'CurriculumExample',
    'DifficultyLevel'
]
