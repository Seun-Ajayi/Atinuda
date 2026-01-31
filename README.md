# Àtinúdá

Àtinúdá is a Yorùbá word that literally means *creative*, but contextually means “to birth from within.”
This meaning motivates the repository: we explores reflective, automatic prompt optimization, where prompts are not rewritten externally, but *improve themselves from within* through structured feedback.

This repository contains the scripts and experimental code used in our study:

> “What Do Prompts Reveal About Model Capabilities in Low-Resource Languages?”

We apply GEPA (Genetic-Pareto) to evaluate how reflective prompt optimization surfaces latent model capabilities, particularly in African low-resource languages.

### Tasks Covered
We evaluate across multiple task families, including:
- AfriQA – Extractive Question Answering
- Belebele – Multiple-choice Reading Comprehension
- AfriMMLU – Knowledge-intensive Multiple-choice QA
- PIQA – Physical Commonsense Reasoning
- AfriDocMT – Machine Translation
  
We explore prompt optimization in two regimes:
- Cross-Model Reflection, where a strongermodel provides feedback to improve a weaker executor, and
- Self Reflection, where a single model serves as both executor and reflection model.
  
Àtinúdá is both:
- an experimental artifact supporting the paper’s results, and
- a research sandbox for studying reflective optimization, prompt efficiency, and low-resource NLP evaluation.
