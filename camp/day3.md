# day3

# Evals
Notebooks
- personal benchmark (recommended) https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/workshops/personal_benchmark/personal_benchmark.ipynb

- evals hard, https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/workshops/evals/evals_hard.ipynb
- evals normal (diving into the Inspect API to create evals) https://colab.research.google.com/github/EffiSciencesResearch/ML4G-2.0/blob/master/workshops/evals/evals_normal.ipynb

### Why do evals 
- Capabilities detection
  - we can't predict capabilities before training
  - no other way to determine what AI system can do
- Key inputs
  - Uk and US Safety institutes
  - OPen AI preparedness framework
  - Anthropic Responsible scaling policies
  - Deepmind Frontier Safety Framework

### conceptns
#### bits of help

*end-to-end -> milestone -> expert best of N -> Golden solution*

- **End-to-end**

    Black-box: run task → script checks output

  - fully automated, fast
  - misses edge-cases, unreliable for hard tasks

- **Milestone**

  Split into steps (e.g. parse → pseudo-code → run code)

  - pinpoints failure stage
  - need extra tests/tools per step

- **Expert best-of-N**

    Human reviews N outputs and picks the best

  - catches style/semantic nuances
  - slow, costly, requires N generations

- **Golden solution**

    Compare model output to gold reference(s) (exact/F1/BLEU…)

  - most accurate for fixed-answer tasks
  -  needs pre-made gold examples, limited scope

### Paper: Observational scaling laws and the predictability of LM performance
- https://arxiv.org/abs/2405.10938
- Observational scaling laws = fit curves without retraining; mine scores of ≈ 100 public LMs spanning 5 orders of compute.
- Benchmarks (MMLU, GSM-8K, HumanEval…) mapped into a 3-D PCA “capability space”; log-compute is nearly linear inside each family, families differ by a constant efficiency shift.
- Simple log-linear / sigmoid per metric predicts emergent knees (GSM-8K, BBH) and GPT-4 agent scores within ± 3 pp using only sub-7B checkpoints.
- Same framework forecasts CoT / Self-Consistency gains; trick benefits fade beyond ≈ 75 % raw accuracy.
- Applications: cheap forward forecasts (e.g., Mixtral-8 × 22 B, GPT-4), minimal model subsets for new benchmark design, clearer risk timelines.
- Limits: depends on public checkpoints & noisy compute estimates; PCs are benchmark-specific; big algorithmic jumps can break the constant-shift assumption.

### Benchmark: The Weapons of Mass Destruction proxy benchmark

### METR: Measuring AI Ability to Complete Long Tasks
- https://www.alignmentforum.org/posts/deesrjitvXM4xYGZd/metr-measuring-ai-ability-to-complete-long-tasks

### Towards understanding-based safety evaluations
- https://www.alignmentforum.org/posts/uqAdqrvxqGqeBHjTP/towards-understanding-based-safety-evaluations


- Evan Hubinger
> My concern is that, in such a situation, being able to robustly evaluate the safety of a model could be a more difficult problem than finding training processes that robustly produce safe models.

## Personal Benchmark with Google Sheets
- https://colab.research.google.com/drive/1mfeHpblnpV58b5koXhgp-RjbIkpBhNbH
- 
## Evals with Inspect
- https://colab.research.google.com/drive/1_sUBU21LRQHlQ2-1mS1aRmJsh3zmyHF4
- https://colab.research.google.com/drive/1QYSBfIdsdTfa715Eo1ciclGU2f_u7DTB#scrollTo=0EMtP-wd4T3k
- https://colab.research.google.com/drive/1QYSBfIdsdTfa715Eo1ciclGU2f_u7DTB#scrollTo=9pe76ptf4T3d
## Recomendations 
- Igor Ivanov (Oxford Research group)
  - Here’s a beginner’s guide to evals: https://www.lesswrong.com/posts/2PiawPFJeyCQGcwXG/a-starter-guide-for-evals
  - And here’s a list of good evals project ideas for newcomers: https://www.lesswrong.com/posts/LhnqegFoykcjaXCYH/100-concrete-projects-and-open-problems-in-evals
- Elena Ericheva (METR)
  - 1
  - 2

# Strategies and first steps

### Rough plans:
#### Theory of AI 
- Move AI safety beyond the pre-paradigmatic stage: methodological and axiomatic research (MAR)
- And close that strang gap between ML and AI safety by shaping public opinion and awareness
#### Research frontier
- Establish more (+20-40) laboratories with diverse agendas in AI safety and AI theory (NOT just another team training its own LLM in hopes of reaching AGI first);
#### Governanse and public com
- Ensure international coordination and regulation to shape public opinion, raise awareness, and enable mutual oversight.
### Next steps:
#### Theory of AI: 
- workshop on evals reproduction in dec 2025, workshop on MAR in feb 2026, updates or scaling on previous (dec 2026)
#### Research: 
- Test config space hypothesis on hybrid architectures
- Network with people who are interested in creating new agenda

### Important assumptions:
- No single current approach is sufficient to solve the technical safety problem.
- Making safety post-pre-paradigmatic will help the field to produce results faster.
- Programmes such as MATS, Arena and ML4Good are effective at producing new AI-safety researchers.
- Effective Governance approach exists and at least some people are already practising it.
- Diverse labs will pivot when their agenda proves wrong or sub-optimal.
- The assumption that exists but i do not notice it =)


## my thoughts and questions 
- Is there a benchmark comparing a model’s ability to solve end-to-end tasks versus solving those same tasks broken down into milestones?
- Are there evaluation results (progress) for the same model family as they moved from golden solutions toward the right-hand methods?
- **Are these stages truly ordered by difficulty in this exact sequence? How to design an experiment to solve it?**
- Are there benchmarks that measure how a model’s capabilities depend on the Kolmogorov complexity of a task?
- Can this be measured in practice somehow?
- What other complexity classes can we use to evaluate model capabilities?
- what governance initiative would work
