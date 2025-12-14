"""
AfriMMLU task implementation for DSPy evaluation pipeline.
"""
from typing import List, Tuple

import dspy
from datasets import load_dataset

from base import BaseTask, _parse_str_list


class AfriMMLUTask(BaseTask):
    """AfriMMLU multiple choice task."""
    
    def __init__(self, model_name: str, lang: str, api_key: str, **kwargs):
        super().__init__(model_name, lang, api_key, **kwargs)
        self.dataset_name = "masakhane/afrimmlu"
    
    def init_dataset(self) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Load AfriMMLU dataset for specified language."""
        dataset = load_dataset(self.dataset_name, self.lang)
        
        train_split = dataset["validation"]
        validation_split = dataset["dev"]
        test_split = dataset["test"]
        
        train_split = [
            dspy.Example({
                "question": x['question'],
                "subject": x['subject'],
                'choices': _parse_str_list(x['choices']),
                "answer": x['answer'],
            }).with_inputs("question", "subject", "choices")
            for x in train_split
        ]
        
        validation_split = [
            dspy.Example({
                "question": x['question'],
                "subject": x['subject'],
                'choices': _parse_str_list(x['choices']),
                "answer": x['answer'],
            }).with_inputs("question", "subject", "choices")
            for x in validation_split
        ]
        
        test_split = [
            dspy.Example({
                "question": x['question'],
                "subject": x['subject'],
                'choices': _parse_str_list(x['choices']),
                "answer": x['answer'],
            }).with_inputs("question", "subject", "choices")
            for x in test_split
        ]
        
        train_set = train_split
        val_set = validation_split
        test_set = test_split
        
        return train_set, val_set, test_set
    
    def create_program(self) -> dspy.Module:
        """Create AfriMMLU program."""
        class GenerateResponse(dspy.Signature):
            """Answer the question with the correct option (A–D) from the given choices."""
            question = dspy.InputField()
            subject = dspy.InputField()
            choices = dspy.InputField(desc="List of four answer options ordered as [A,B,C,D].")
            answer = dspy.OutputField(
                desc="Return exactly one letter in {A,B,C,D}, where A=choices[0], B=choices[1], C=choices[2], D=choices[3]."
            )
        
        return dspy.ChainOfThought(GenerateResponse)
    
    def metric(self, example: dspy.Example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
        """Check if predicted letter matches correct answer."""
        correct_answer = str(example["answer"]).strip()
        llm_answer = str(prediction.answer).strip().upper()
        
        if llm_answer not in {"A", "B", "C", "D"}:
            return 0
        
        return int(llm_answer == correct_answer)
    
    def metric_with_feedback(
        self, 
        example: dspy.Example, 
        prediction, 
        trace=None, pred_name=None, pred_trace=None
    ) -> dspy.Prediction:
        """Metric with feedback for optimization."""
        letters = ["A", "B", "C", "D"]
        
        correct_letter = str(example["answer"]).strip()
        choices = example["choices"]
        correct_idx = letters.index(correct_letter)
        correct_choice = choices[correct_idx]
        
        llm_answer = str(prediction.answer).strip().upper()
        pred_letter = llm_answer.upper()
        
        # Invalid prediction (not A–D)
        if pred_letter not in letters:
            feedback_text = (
                f"The final answer must be one of A, B, C, or D. "
                f"You responded with '{llm_answer}', which is not a valid option. "
                f"The correct answer is '{correct_choice}', which is option {correct_letter}."
            )
            return dspy.Prediction(score=0, feedback=feedback_text)
        
        # Exact match on letter
        score = int(pred_letter == correct_letter)
        
        if score == 1:
            feedback_text = (
                f"Your answer is correct. The correct answer is "
                f"'{correct_choice}', which is option {correct_letter}."
            )
        else:
            feedback_text = (
                f"Your answer is incorrect. The correct answer is "
                f"'{correct_choice}', which is option {correct_letter}."
            )
        
        return dspy.Prediction(score=score, feedback=feedback_text)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run AfriMMLU evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., yor, hau, swa)")
    parser.add_argument("--output-dir", type=str, default="./results/afrimmlu", help="Output directory")
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Create task and run pipeline
    task = AfriMMLUTask(
        model_name=args.model,
        lang=args.lang,
        api_key=api_key
    )
    
    output_dir = f"{args.output_dir}/{args.model.replace('/', '-')}/{args.lang}"
    summary = task.run_full_pipeline(output_dir)
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
