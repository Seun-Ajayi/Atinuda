"""
Belebele task implementation for DSPy evaluation pipeline.
"""
from typing import List, Tuple
import dspy
from datasets import load_dataset, DatasetDict
from base import BaseTask


class BelebeleTask(BaseTask):
    """Belebele multiple choice task."""
    
    def __init__(self, model_name: str, lang: str, api_key: str, **kwargs):
        super().__init__(model_name, lang, api_key, **kwargs)
        self.dataset_name = "facebook/belebele"
    
    def preprocess_dataset(self, dataset_dict: DatasetDict):
        """
        Merge mc_answer1/mc_answer2/mc_answer3/mc_answer4 into a single list field `answers`.
        Drops the original mc_answer1, mc_answer2, mc_answer3 and mc_answer4 fields.
        """

        def transform(batch):
            # create the new list field
            s1 = batch["mc_answer1"] if batch["mc_answer1"] is not None else "<NO_ANSWER>"
            s2 = batch["mc_answer2"] if batch["mc_answer2"] is not None else "<NO_ANSWER>"
            s3 = batch["mc_answer3"] if batch["mc_answer3"] is not None else "<NO_ANSWER>"
            s4 = batch["mc_answer4"] if batch["mc_answer4"] is not None else "<NO_ANSWER>"


            batch["answers"] = [s1, s2, s3, s4]

            del batch["mc_answer1"]
            del batch["mc_answer2"]
            del batch["mc_answer3"]
            del batch["mc_answer4"]

            return batch

        # apply transformation to the entire test split
        ds = dataset_dict["test"].map(transform)

        return DatasetDict({
            "test": ds,
        })
    
    def init_dataset(self) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Load PIQA dataset for specified language."""
        dataset = load_dataset(self.dataset_name, self.lang)
        dataset = self.preprocess_dataset(dataset)
        dataset = dataset["test"]
        dataset = dataset.shuffle(seed=42)

        train_split = dataset.select(range(0, 50))
        validation_split = dataset.select(range(50, 100))
        test_split = dataset.select(range(100, 500))
        train_split = [
            dspy.Example({
                "passage": x['flores_passage'],
                "question": x['question'],
                "answer_list": x['answers'],
                "correct_answer_number": x['correct_answer_num'],
            }).with_inputs("passage", "question", "answer_list")
            for x in train_split
        ]

        validation_split = [
            dspy.Example({
                "passage": x['flores_passage'],
                "question": x['question'],
                "answer_list": x['answers'],
                "correct_answer_number": x['correct_answer_num'],
            }).with_inputs("passage", "question", "answer_list")   
            for x in validation_split
        ]    

        test_split = [
            dspy.Example({
                "passage": x['flores_passage'],
                "question": x['question'],
                "answer_list": x['answers'],
                "correct_answer_number": x['correct_answer_num'],
            }).with_inputs("passage", "question", "answer_list")     
            for x in test_split
        ]

        train_set = train_split
        val_set = validation_split
        test_set = test_split

        return train_set, val_set, test_set
    
    def create_program(self) -> dspy.Module:
        """Create Belebele program."""
        class ReadingComprehension(dspy.Signature):
            """Answer a multiple-choice question using the passage."""
            question = dspy.InputField(desc="The question to answer from the passage.")
            passage = dspy.InputField(desc="The passage that contains the answer.")
            answer_list = dspy.InputField(desc="List of answer options.")
            correct_answer_number = dspy.OutputField(desc="Return the 1-based index of the correct option.")

        program = dspy.ChainOfThought(ReadingComprehension)
    
    def metric(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        correct_answer = str(example["correct_answer_number"]).strip()
        llm_answer = str(prediction.correct_answer_number).strip()

        if llm_answer not in {"1", "2", "3", "4"}:
            return 0

        return int(llm_answer == correct_answer)

    def metric_with_feedback(self, example, prediction, trace=None, pred_name=None, pred_trace=None):
        answers = example["answer_list"]
        gold = str(example["correct_answer_number"]).strip()
        passage = example.get("passage", "")

        pred = str(getattr(prediction, "correct_answer_number", "")).strip()

        valid = {str(i) for i in range(1, len(answers) + 1)}

        if pred not in valid:
            gold_idx = int(gold) - 1
            feedback_text = (
                f"You are to return a single option number in {sorted(valid)}."
                f"You returned '{pred}' which is not a valid option. "
                f"The correct option is `{gold}` in the answer list, which corresponds to: {answers[gold_idx]}."
            )
            if passage:
                feedback_text += (
                f"\n\n Here is the passage that contains the correct answer:\n{passage}\n\n"
                "Next time: locate the sentence that directly answers the question, then map it to the option list."
            )        
            return dspy.Prediction(score=0, feedback=feedback_text)

        score = int(pred == gold)
        gold_idx = int(gold) - 1
        pred_idx = int(pred) - 1

        if score == 1:
            feedback_text = (
                f"Your answer is correct! It is option `{gold}` in the answer list, "
                f"which corresponds to: `{answers[gold_idx]}`."
            )
        else:
            feedback_text = (
                f"Your answer is incorrect! "
                f"The correct option is `{gold}` in the answer list, which corresponds to: `{answers[gold_idx]}`."
            )

        if passage:
            feedback_text += (
            f"\n\n Here is the passage that contains the correct answer:\n{passage}\n\n"
            "Next time: locate the sentence that directly answers the question, then map it to the option list."
        )

        return dspy.Prediction(score=score, feedback=feedback_text)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run Belebele evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., openai/gpt-4.1-mini)")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., yor, hau, swa)")
    parser.add_argument("--output-dir", type=str, default="./results/Belebele", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=32000, help="Maximum Tokens")
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    # Create task and run pipeline
    task = BelebeleTask(
        model_name=args.model,
        lang=args.lang,
        api_key=api_key,
        max_tokens=args.max_tokens
    )
    
    output_dir = args.output_dir
    summary = task.run_full_pipeline(output_dir)
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
