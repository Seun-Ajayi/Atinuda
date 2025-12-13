"""
PIQA task implementation for DSPy evaluation pipeline.
"""
from typing import List, Tuple

import dspy
from datasets import load_dataset, DatasetDict

from base import BaseTask, _parse_str_list


class PIQATask(BaseTask):
    """PIQA multiple choice task."""
    
    def __init__(self, model_name: str, lang: str, api_key: str, **kwargs):
        super().__init__(model_name, lang, api_key, **kwargs)
        self.dataset_name = "masakhane/piqa_yoruba_pidgin"

    def preprocess_dataset(self, dataset_dict: DatasetDict):
        """
        Split dataset into two (Yoruba, Nigerian Pidgin) and
        merge solution0/solution1 into a single list field `solutions`.
        Drops the original solution0 and solution1 fields.
        """

        def transform(batch):
            # create the new list field
            s0 = batch["solution0"] if batch["solution0"] is not None else "<NO_ANSWER>"
            s1 = batch["solution1"] if batch["solution1"] is not None else "<NO_ANSWER>"

            batch["solutions"] = [s0, s1]

            del batch["solution0"]
            del batch["solution1"]
            return batch

        # apply transformation to the entire train split
        ds = dataset_dict["train"].map(transform)

        # split by language
        yoruba_ds = ds.filter(lambda x: x["language"] == "Yoruba")
        pidgin_ds = ds.filter(lambda x: x["language"] == "Nigerian Pidgin")

        return DatasetDict({
            "yoruba": yoruba_ds,
            "pidgin": pidgin_ds,
        })
    
    def init_dataset(self) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Load PIQA dataset for specified language."""
        dataset = load_dataset(self.dataset_name)
        dataset = self.preprocess_dataset(dataset)
        dataset = dataset[self.lang]
        dataset = dataset.shuffle(seed=42)

        train_split = dataset.select(range(0, 50))
        validation_split = dataset.select(range(50, 100))
        test_split = dataset.select(range(100, 200))
        train_split = [
            dspy.Example({
                "goal": x['goal'],
                "solutions": x['solutions'],
                "label": x['label'],
            }).with_inputs("goal", "solutions")
            for x in train_split
        ]

        validation_split = [
            dspy.Example({
                "goal": x['goal'],
                "solutions": x['solutions'],
                "label": x['label'],
            }).with_inputs("goal", "solutions")      
            for x in validation_split
        ]    

        test_split = [
            dspy.Example({
                "goal": x['goal'],
                "solutions": x['solutions'],
                "label": x['label'],
            }).with_inputs("goal", "solutions")       
            for x in test_split
        ]

        train_set = train_split
        val_set = validation_split
        test_set = test_split

        return train_set, val_set, test_set
    
    def create_program(self) -> dspy.Module:
        """Create PIQA program."""
        class GenerateResponse(dspy.Signature):
            """Pick the better solution (0 or 1) for the given goal."""
            goal = dspy.InputField(
                desc="Natural-language description of the goal or intent to achieve."
            )
            solutions = dspy.InputField(
                desc="List [solution0, solution1] with two candidate ways to achieve the goal."
            )
            label = dspy.OutputField(
                desc="Return '0' if solution0 is better, '1' if solution1 is better."
            )
        
        return dspy.ChainOfThought(GenerateResponse)
    
    def metric(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        correct_answer = str(example["label"]).strip()
        llm_answer = str(prediction.label).strip()

        if llm_answer not in {"0", "1"}:
            return 0

        return int(llm_answer == correct_answer)
    
    def metric_with_feedback(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        valid_labels = {"0", "1"}

        gold_label = str(example["label"]).strip()
        solutions  = example["solutions"]
        correct_idx = int(gold_label)
        correct_solution = solutions[correct_idx]

        llm_answer = str(prediction.label).strip()
        pred_label = llm_answer

        # Invalid prediction (not "0" or "1")
        if pred_label not in valid_labels:
            feedback_text = (
                f"The final answer must be either '0' or '1'. "
                f"You responded with '{llm_answer}', which is not a valid option. "
                f"The correct label is '{gold_label}', which corresponds to: {correct_solution}"
            )
            return dspy.Prediction(score=0, feedback=feedback_text)

        # Exact match on label
        score = int(pred_label == gold_label)

        if score == 1:
            feedback_text = (
                f"Your answer is correct. The correct label is '{gold_label}', "
                f"which corresponds to solutions[{correct_idx}]: {correct_solution}"
            )
        else:
            feedback_text = (
                f"Your answer is incorrect. The correct label is '{gold_label}', "
                f"which corresponds to solutions[{correct_idx}]: {correct_solution}"
            )

        return dspy.Prediction(score=score, feedback=feedback_text)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run PIQA evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., yor, hau, swa)")
    parser.add_argument("--output-dir", type=str, default="./results/PIQA", help="Output directory")
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Create task and run pipeline
    task = PIQATask(
        model_name=args.model,
        lang=args.lang,
        api_key=api_key
    )
    
    output_dir = f"{args.output_dir}/{args.model.replace('/', '-')}/{args.lang}"
    summary = task.run_full_pipeline(output_dir)
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
