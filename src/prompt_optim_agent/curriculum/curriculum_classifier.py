"""
Curriculum Learning Classifier

Classifies training examples into Easy/Medium/Hard based on:
1. Semantic complexity (embedding distance from centroid)
2. Baseline performance (initial prompt success/failure)

"""

import numpy as np
from typing import List, Dict, Tuple, Literal
from dataclasses import dataclass
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from ..test_helper import eval_instruction_with_loader
from sentence_transformers import SentenceTransformer

DifficultyLevel = Literal['EASY', 'MEDIUM', 'HARD']


@dataclass
class CurriculumExample:
    """Enriched example with curriculum metadata."""
    question: str
    answer: str
    difficulty: DifficultyLevel
    is_semantic_outlier: bool
    baseline_failed: bool
    semantic_distance: float
    original_index: int


class CurriculumClassifier:
    """
    Pre-processes dataset to classify examples by difficulty.
    
    Uses two signals:
    1. Semantic Complexity: Distance from embedding centroid
    2. Baseline Performance: Success/failure with initial prompt
    
    Classification Logic:
    - EASY: Model succeeds + Common pattern
    - MEDIUM: Model fails on common OR succeeds on outlier
    - HARD: Model fails + Outlier pattern
    """
    
    def __init__(
        self,
        task,
        base_model,
        logger: logging.Logger = None,
        semantic_outlier_percentile: float = 0.80,  # Top 20% distances
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize Curriculum Classifier.
        
        Args:
            task: Task instance with dataset and evaluation methods
            base_model: Model for baseline evaluation
            logger: Logger instance
            semantic_outlier_percentile: Percentile threshold for outlier detection
            embedding_model: HuggingFace model for embeddings
        """
        self.task = task
        self.base_model = base_model
        self.logger = logger or logging.getLogger(__name__)
        self.semantic_outlier_percentile = semantic_outlier_percentile
        self.embedding_model_name = embedding_model
        
        # Will be populated during classification
        self.embeddings = None
        self.centroid = None
        self.distances = None
        self.distance_threshold = None
        
    def _log(self, message: str, level: str = "info"):
        """Centralized logging."""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def calculate_semantic_complexity(
        self, 
        examples: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate semantic complexity using embeddings.
        
        Args:
            examples: List of {'question': str, 'answer': str}
            
        Returns:
            embeddings: (N, D) array of embeddings
            distances: (N,) array of distances to centroid
            threshold: Distance threshold for outlier detection
        """
        self._log("="*80)
        self._log("STEP 1: SEMANTIC COMPLEXITY ANALYSIS")
        self._log("="*80)

        questions = [ex['question'] for ex in examples]
        
        self._log(f"Generating embeddings for {len(questions)} examples...")
        self._log(f"Using model: {self.embedding_model_name}")

        try:
            model = SentenceTransformer(self.embedding_model_name)
            embeddings = model.encode(
                questions, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
        except ImportError:
            self._log(
                "Encode failed", 
                level="error"
            )
            raise
        
        self._log(f"Embeddings shape: {embeddings.shape}")

        centroid = embeddings.mean(axis=0)
        self._log(f"Centroid computed: shape {centroid.shape}")
        
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        self._log(f"Distance statistics:")
        self._log(f"  Min: {distances.min():.4f}")
        self._log(f"  Max: {distances.max():.4f}")
        self._log(f"  Mean: {distances.mean():.4f}")
        self._log(f"  Std: {distances.std():.4f}")
        
        # Calculate threshold (top 20% are outliers)
        threshold = np.percentile(distances, self.semantic_outlier_percentile * 100)
        outlier_count = (distances >= threshold).sum()
        
        self._log(f"Outlier threshold (p{self.semantic_outlier_percentile*100:.0f}): {threshold:.4f}")
        self._log(f"Semantic outliers: {outlier_count}/{len(distances)} ({outlier_count/len(distances)*100:.1f}%)")
        
        # Store for later use
        self.embeddings = embeddings
        self.centroid = centroid
        self.distances = distances
        self.distance_threshold = threshold
        
        return embeddings, distances, threshold
    
    def evaluate_baseline_performance(
        self, 
        examples: List[Dict],
        initial_prompt: str
    ) -> np.ndarray:
        """
        Evaluate baseline performance with initial prompt.
        
        Args:
            examples: List of {'question': str, 'answer': str}
            initial_prompt: prompt inicial
            
        Returns:
            failed_mask: Boolean array where True = failed
        """
        self._log("\n" + "="*80)
        self._log("STEP 2: BASELINE  EVALUATION")
        self._log("="*80)
        self._log(f"Initial prompt: {initial_prompt}")
        self._log(f"Evaluating {len(examples)} examples...")
        
        # Create temporary dataloader
        
        
        temp_dataset = self.task.TaskDataset(examples)
        temp_loader = DataLoader(
            temp_dataset, 
            batch_size=8,
            shuffle=False
        )
        
        # Evaluate
        metric, eval_output = eval_instruction_with_loader(
            task=self.task,
            eval_prompt=initial_prompt,
            base_model=self.base_model,
            dataloader=temp_loader,
            temperature=0.0,
            record_outputs=True
        )
        
        # Extract success/failure
        correct_mask = np.array(eval_output['correct'], dtype=bool)
        failed_mask = ~correct_mask
        
        success_count = correct_mask.sum()
        failure_count = failed_mask.sum()
        
        self._log(f"Baseline Performance:")
        self._log(f"  Metric: {metric:.4f}")
        self._log(f"  Success: {success_count}/{len(examples)} ({success_count/len(examples)*100:.1f}%)")
        self._log(f"  Failure: {failure_count}/{len(examples)} ({failure_count/len(examples)*100:.1f}%)")
        
        return failed_mask
    
    def bucket_dataset(
        self,
        examples: List[Dict],
        semantic_outlier_mask: np.ndarray,
        baseline_failed_mask: np.ndarray
    ) -> List[CurriculumExample]:
        """
        Classify examples into EASY/MEDIUM/HARD buckets.
        
        Classification Logic:
        - EASY: Success + Common (not outlier, not failed)
        - MEDIUM: (Fail + Common) OR (Success + Outlier)
        - HARD: Fail + Outlier
        
        Args:
            examples: Original examples
            semantic_outlier_mask: Boolean array (True = outlier)
            baseline_failed_mask: Boolean array (True = failed)
            
        Returns:
            List of CurriculumExample with difficulty labels
        """
        self._log("\n" + "="*80)
        self._log("STEP 3: DIFFICULTY BUCKETING")
        self._log("="*80)
        
        curriculum_examples = []
        
        for idx, example in enumerate(examples):
            is_outlier = semantic_outlier_mask[idx]
            is_failed = baseline_failed_mask[idx]
            
            # Classification logic
            if not is_failed and not is_outlier:
                difficulty = 'EASY'
            elif (is_failed and not is_outlier) or (not is_failed and is_outlier):
                difficulty = 'MEDIUM'
            else:  # is_failed and is_outlier
                difficulty = 'HARD'
            
            curriculum_examples.append(
                CurriculumExample(
                    question=example['question'],
                    answer=example['answer'],
                    difficulty=difficulty,
                    is_semantic_outlier=bool(is_outlier),
                    baseline_failed=bool(is_failed),
                    semantic_distance=float(self.distances[idx]),
                    original_index=idx
                )
            )
        
        # Statistics
        easy_count = sum(1 for ex in curriculum_examples if ex.difficulty == 'EASY')
        medium_count = sum(1 for ex in curriculum_examples if ex.difficulty == 'MEDIUM')
        hard_count = sum(1 for ex in curriculum_examples if ex.difficulty == 'HARD')
        total = len(curriculum_examples)
        
        self._log(f"Difficulty Distribution:")
        self._log(f"  EASY:   {easy_count:3d} ({easy_count/total*100:.1f}%)")
        self._log(f"  MEDIUM: {medium_count:3d} ({medium_count/total*100:.1f}%)")
        self._log(f"  HARD:   {hard_count:3d} ({hard_count/total*100:.1f}%)")
        self._log(f"  TOTAL:  {total:3d}")
        
        # Detailed breakdown
        self._log("\nDetailed Breakdown:")
        self._log(f"  EASY   = Success ∩ Common:     {easy_count}")
        self._log(f"  MEDIUM = (Fail ∩ Common) ∪     ")
        self._log(f"           (Success ∩ Outlier):  {medium_count}")
        self._log(f"  HARD   = Fail ∩ Outlier:       {hard_count}")
        
        return curriculum_examples
    
    def classify_dataset(
        self,
        examples: List[Dict],
        initial_prompt: str
    ) -> List[CurriculumExample]:
        """
        Full pipeline: classify dataset into difficulty levels.
        
        Args:
            examples: List of {'question': str, 'answer': str}
            initial_prompt: Initial prompt for baseline evaluation
            
        Returns:
            List of CurriculumExample with difficulty labels
        """
        self._log("\n" + "==="*40)
        self._log("CURRICULUM LEARNING CLASSIFICATION")
        self._log("==="*40 + "\n")
        
        # Step 1: Semantic complexity
        embeddings, distances, threshold = self.calculate_semantic_complexity(examples)
        semantic_outlier_mask = distances >= threshold
        
        # Step 2: Baseline performance
        baseline_failed_mask = self.evaluate_baseline_performance(examples, initial_prompt)
        
        # Step 3: Bucket into difficulties
        curriculum_examples = self.bucket_dataset(
            examples,
            semantic_outlier_mask,
            baseline_failed_mask
        )
        
        self._log("\n" + "="*80)
        self._log("Classificação completaaa")
        self._log("="*80 + "\n")
        
        return curriculum_examples
