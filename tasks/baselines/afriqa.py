
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
    q_key = "question" if mode == "native" else "translated_question"
    a_key = "answers"  if mode == "native" else "translated_answer"

    def prompt_fn(line: dict, task_name: str = None):
        q = (line.get(q_key) or "").strip()
        golds_raw = line.get(a_key)

        if not q or not golds_raw:
            return None

        golds = _parse_str_list(golds_raw)
        if not golds:
            return None

        return Doc(
            task_name=task_name,
            query=f"Answer in the same language as the question.\n\nQuestion: {q}\nAnswer: ",
            choices=golds,
            gold_index=list(range(len(golds))),
        )

    return prompt_fn

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"afriqa:{lang}:{mode}",
        prompt_function=make_prompt(mode),
        hf_repo="masakhane/afriqa",
        hf_subset=lang,
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        generation_size=64,
        stop_sequence=["\n", "Question:", "question:"],
        metrics=[Metrics.exact_match, Metrics.f1_score],
        version=0,
        num_samples=20,
    )
    for lang in afriqa_languages
    for mode in ["native"]
]
