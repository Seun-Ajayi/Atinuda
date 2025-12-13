"""
AfriQA task implementation for DSPy evaluation pipeline.
"""
from typing import List, Tuple, Dict
import re
import unicodedata
from collections import Counter

import dspy
from datasets import load_dataset

from base import BaseTask, _parse_str_list


class AfriQATask(BaseTask):
    """AfriQA multiple choice task."""
    
    def __init__(self, model_name: str, lang: str, api_key: str, **kwargs):
        super().__init__(model_name, lang, api_key, **kwargs)
        self.dataset_name = "masakhane/afriqa-gold-passages"
    
    def init_dataset(self) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Load AfriQA dataset for specified language."""
        dataset = load_dataset(self.dataset_name, self.lang)
        
        train_split = dataset["train"]
        test_split = dataset["test"]
        
        train_split = [
            dspy.Example({
                "question_lang": x['question_lang'],
                'context': x['context'],
                'answer_lang': _parse_str_list(x['answer_lang']),
            }).with_inputs("question_lang", "context")
            for x in train_split
        ]

        import random
        random.Random(0).shuffle(train_split)
        train_split = train_split[:100]
        tot_num = len(train_split)

        
        test_split = [
            dspy.Example({
                "question_lang": x['question_lang'],
                'context': x['context'],
                'answer_lang': _parse_str_list(x['answer_lang']),
            }).with_inputs("question_lang", "context")
            for x in test_split
        ]
        
        train_set = train_split[:int(0.5 * tot_num)]
        val_set = train_split[int(0.5 * tot_num):]
        test_set = test_split
        
        return train_set, val_set, test_set
    
    def create_program(self) -> dspy.Module:
        """Create AfriQA program."""
        class GenerateResponse(dspy.Signature):
            """Answer the question using the context provided."""
            question_lang = dspy.InputField()
            context  = dspy.InputField()
            answer_lang   = dspy.OutputField(desc="Short final answer only")
        
        return dspy.ChainOfThought(GenerateResponse)
    
    @staticmethod
    def _strip_accents(cls, s: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFD", s)
                    if unicodedata.category(ch) != "Mn")

    @classmethod
    def _normalize(cls, s: str) -> str:
        s = (s or "").strip().lower()
        s = cls._strip_accents(s)
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _token_f1(cls, pred: str, gold: str) -> float:
        p = cls._normalize(pred).split()
        g = cls._normalize(gold).split()
        if not p and not g: return 1.0
        if not p or not g: return 0.0
        common = Counter(p) & Counter(g)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        prec = num_same / len(p)
        rec  = num_same / len(g)
        return (2 * prec * rec) / (prec + rec)
    
    def metric_em(self, example, prediction, **_):
        golds = example.get("answer_lang")
        pred  = getattr(prediction, "answer_lang", "")
        pn = self._normalize(pred)
        return float(any(pn == self._normalize(g) for g in golds))

    def metric_f1(self, example, prediction, **_):
        golds = example.get("answer_lang")
        pred  = getattr(prediction, "answer_lang", "")
        return max((self._token_f1(pred, g) for g in golds), default=0.0)
        
    def evaluate_model(
        self,
        program: dspy.Module,
        test_set: List[dspy.Example],
        save_prefix: str
    ) -> Dict[str, float]:
        """Evaluate on both EM and F1 metrics."""
        print(f"\n{'='*60}")
        print(f"Evaluating: {save_prefix}")
        print(f"{'='*60}\n")
        
        # Evaluate with EM
        print("Evaluating with Exact Match (EM)...")
        evaluate_em = dspy.Evaluate(
            devset=test_set,
            metric=self.metric_em,
            num_threads=32,
            display_table=False,
            display_progress=True
        )
        em_result = evaluate_em(program, save_as_json=f"{save_prefix}_em.json")
        em_score = em_result.score
        print(f"EM Score: {em_score:.2f}%")
        
        # Evaluate with F1
        print("\nEvaluating with F1 Score...")
        evaluate_f1 = dspy.Evaluate(
            devset=test_set,
            metric=self.metric_f1,
            num_threads=32,
            display_table=False,
            display_progress=True
        )
        f1_result = evaluate_f1(program, save_as_json=f"{save_prefix}_f1.json")
        f1_score = f1_result.score
        print(f"F1 Score: {f1_score:.2f}%")
        
        return {
            "em": em_score,
            "f1": f1_score
        }
        

    def metric_with_feedback(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        golds = example.get("answer_lang", [])
        if isinstance(golds, str):
            golds = [golds]
        golds = [g.strip() for g in golds if isinstance(g, str) and g.strip()]

        context = example.get("context", "")

        pred = getattr(prediction, "answer_lang", "")
        pred = (pred or "").strip()

        if not pred:
            feedback_text = (
                "The final answer must be a non-empty short string and nothing else. "
                f"You responded with {pred!r}."
            )
            if golds:
                feedback_text += f" The correct exact-match answer is one of: {golds}."
            if context:
                feedback_text += (
                    f"\n\nHere's some context useful in answering the question:\n{context}\n\n"
                    "Think about what takeaways you can learn from this context to improve your future answers "
                    "and approach to similar question."
                )
            return dspy.Prediction(score=0.0, feedback=feedback_text)

        score = float(any(pred == g for g in golds))

        if score == 1.0:
            feedback_text = f"Your answer is correct. The correct answer is '{pred}'."
        else:
            feedback_text = (
                "Your answer is incorrect. The correct and exact-match answer is "
                f"one of: {golds}."
            )

        if context:
            feedback_text += (
                f"\n\nHere's some context useful in answering the question:\n{context}\n\n"
                "Think about what takeaways you can learn from this context to improve your future answers "
                "and approach to similar question."
            )

        return dspy.Prediction(score=score, feedback=feedback_text)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run AfriQA evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., yor, hau, swa)")
    parser.add_argument("--output-dir", type=str, default="./results/AfriQA", help="Output directory")
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Create task and run pipeline
    task = AfriQATask(
        model_name=args.model,
        lang=args.lang,
        api_key=api_key
    )
    
    output_dir = f"{args.output_dir}/{args.model.replace('/', '-')}/{args.lang}"
    summary = task.run_full_pipeline(output_dir)
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
