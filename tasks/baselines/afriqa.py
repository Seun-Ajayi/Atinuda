import ast
from typing import List
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

import litellm

litellm._turn_on_debug()

afriqa_languages = [
    "yor" # "bem","fon","hau","ibo","kin","swa","twi","wol","yor","zul"
]

PROMPT_PRETEXT = """Answer the question using the context provided."""

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

def make_prompt(mode: str):
    q_key = "question_lang" if mode == "native" else "question_translated"
    a_key = "answer_lang"  if mode == "native" else "translated_answer"
    context_key = "context"

    def prompt_fn(line: dict, task_name: str = None):
        question = line.get(q_key, "").strip()
        gold_answers = line.get(a_key)
        context = line.get(context_key, "")

        if not question or not gold_answers:
            return None

        golds = _parse_str_list(gold_answers)
        if not golds:
            return None

        return Doc(
            task_name=task_name,
            query=f"{PROMPT_PRETEXT}\n\nContext: {context}\nQuestion: {question}\nAnswer: ",
            choices=golds,
            gold_index=list(range(len(golds))),
        )

    return prompt_fn

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"afriqa:{lang}:{mode}",
        prompt_function=make_prompt(mode),
        hf_repo="masakhane/afriqa-gold-passages",
        hf_subset=lang,
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        generation_size=64,
        stop_sequence=["\n", "Question:", "question:"],
        metrics=[Metrics.exact_match, Metrics.f1_score],
        version=0,
        effective_num_docs=5
    )
    for lang in afriqa_languages
    for mode in ["native"]
]