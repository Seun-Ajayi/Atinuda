"""
Base classes and utilities for DSPy evaluation pipeline.
Provides reusable components for different tasks.
"""
import os
import ast
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dspy
from datasets import load_dataset


class BaseTask(ABC):
    """Abstract base class for evaluation tasks."""
    
    def __init__(
        self,
        model_name: str,
        lang: str,
        api_key: str,
        temperature: float = 1.0,
        max_tokens: int = 32000,
        seed: int = 42,
        reflective_model: str = "openai/gpt-5",
    ):
        self.model_name = model_name
        self.lang = lang
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.reflective_model = reflective_model
        
        # Initialize language model
        self.lm = dspy.LM(
            model_name, 
            temperature=temperature, 
            api_key=api_key, 
            max_tokens=max_tokens
        )
        dspy.configure(lm=self.lm)
        
        # Initialize reflection LM for optimization
        self.reflection_lm = dspy.LM(
            model=reflective_model,
            temperature=1.0,
            max_tokens=32000,
            api_key=api_key
        )
    
    @abstractmethod
    def init_dataset(self) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Load and prepare train/val/test splits."""
        pass
    
    @abstractmethod
    def create_program(self) -> dspy.Module:
        """Create the DSPy program/signature."""
        pass
    
    @abstractmethod
    def metric(self, example: dspy.Example, prediction: Any, **kwargs) -> float:
        """Evaluation metric without feedback."""
        pass
    
    @abstractmethod
    def metric_with_feedback(
        self, 
        example: dspy.Example, 
        prediction: Any, 
        **kwargs
    ) -> dspy.Prediction:
        """Evaluation metric with feedback for optimization."""
        pass
    
    def evaluate_model(
        self,
        program: dspy.Module,
        test_set: List[dspy.Example],
        save_path: str
    ) -> float:
        """Evaluate a program on test set."""
        print(f"\n{'='*60}")
        print(f"Evaluating Model: {self.model_name}")
        print(f"{'='*60}\n")
        
        evaluate = dspy.Evaluate(
            devset=test_set,
            metric=self.metric,
            num_threads=32,
            display_table=False,
            display_progress=True
        )
        
        result = evaluate(program, save_as_json=save_path)
        return result.score
    
    def optimize_program(
        self,
        program: dspy.Module,
        train_set: List[dspy.Example],
        val_set: List[dspy.Example]
    ) -> dspy.Module:
        """Optimize program using GEPA."""
        print(f"\n{'='*60}")
        print(f"Optimizing with GEPA")
        print(f"{'='*60}\n")
        
        optimizer = dspy.GEPA(
            metric=self.metric_with_feedback,
            auto="light",
            num_threads=32,
            track_stats=True,
            reflection_minibatch_size=3,
            seed=self.seed,
            reflection_lm=self.reflection_lm
        )
        
        optimized_program = optimizer.compile(
            program,
            trainset=train_set,
            valset=val_set,
        )
        
        return optimized_program
    
    def save_program(self, program: dspy.Module, save_path: str):
        """Save optimized program."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        program.save(save_path, save_program=False)
        print(f"\n✅ Program saved to: {save_path}")
    
    def run_full_pipeline(self, output_dir: str):
        """Run complete evaluation pipeline."""
        # Create output directory structure
        output_dir = Path(output_dir)
        base_dir = output_dir / "base"
        optimized_dir = output_dir / "optimized"
        programs_dir = output_dir / "programs"
        
        # Create all directories
        base_dir.mkdir(parents=True, exist_ok=True)
        optimized_dir.mkdir(parents=True, exist_ok=True)
        programs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        train_set, val_set, test_set = self.init_dataset()
        print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        
        # 1. Create program
        program = self.create_program()
    
        # 2. Evaluate unoptimized model
        unopt_path = base_dir / f"{self.model_name.replace('/', '-')}_{self.lang}.json"
        unopt_score = self.evaluate_model(program, test_set, str(unopt_path))
        print(f"\nUnoptimized Score: {unopt_score}")
        
        # 3. Optimize program
        optimized_program = self.optimize_program(program, train_set, val_set)
        
        # 4. Evaluate optimized model
        opt_path = optimized_dir / f"{self.model_name.replace('/', '-')}_{self.lang}.json"
        opt_score = self.evaluate_model(optimized_program, test_set, str(opt_path))
        print(f"\nOptimized Score: {opt_score}")
        
        # 5. Save optimized program
        program_path = programs_dir / f"{self.model_name.replace('/', '-')}_{self.lang}.json"
        self.save_program(optimized_program, str(program_path))
        
        # 6. Save summary
        summary = {
            "task": self.__class__.__name__,
            "model": self.model_name,
            "language": self.lang,
            "unoptimized": unopt_score,
            "optimized": opt_score,
            "improvement": opt_score - unopt_score,
            "dataset_sizes": {
                "train": len(train_set),
                "val": len(val_set),
                "test": len(test_set)
            }
        }
        
        summary_path = output_dir / f"summary_{self.model_name.replace('/', '-')}_{self.lang}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Unoptimized:       {unopt_score:.2f}%")
        print(f"Optimized:         {opt_score:.2f}%")
        print(f"Improvement:       +{opt_score - unopt_score:.2f}%")
        print(f"{'='*60}\n")
        
        return summary


def _parse_str_list(x) -> List[str]:
    """
    AfriQA stores answers like "['Litunga']" (string). Convert to list[str].
    Also handles default list and plain strings.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(t).strip() for t in v if str(t).strip()]
            except Exception:
                pass
        return [s] if s else []
    return [str(x).strip()] if str(x).strip() else []

