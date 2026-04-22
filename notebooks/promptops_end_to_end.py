# Databricks notebook source
# DBTITLE 1,Setup
# MAGIC %pip install --upgrade "mlflow[databricks]>=3.10.0" openai databricks-openai dspy "gepa>=0.0.26" -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Create Prompts
import os
import mlflow
import pandas as pd
import time
from databricks_openai import DatabricksOpenAI
from mlflow.entities import Feedback

client = DatabricksOpenAI()

# Keep the application model cheap/fast, and use a stronger model for judging.
# Use a GPT-family model for judges/reflection because MLflow prompt optimization
# relies on strict JSON/string outputs from the reflection path.
GENERATION_MODEL = "databricks-gpt-oss-20b"
JUDGE_MODEL_NAME = "databricks-gpt-5-4"
JUDGE_MODEL = f"databricks:/{JUDGE_MODEL_NAME}"

# Small budgets keep this notebook demo-friendly while still showing optimization.
SINGLE_PROMPT_MAX_METRIC_CALLS = 8
MULTI_PROMPT_MAX_METRIC_CALLS = 10

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
import dspy
dspy.configure(num_threads=1)

def extract_text(content) -> str:
    """Extract text from model response — handles reasoning models that
    return structured content (list of parts) instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content if p.get("type") == "text")
    return str(content)


def score_to_float(score) -> float:
    """Normalize scorer outputs so custom objectives work with judge feedback."""
    if isinstance(score, Feedback):
        score = score.value
    if isinstance(score, bool):
        return float(score)
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, str):
        lowered = score.lower()
        if lowered in {"yes", "true"}:
            return 1.0
        if lowered in {"no", "false"}:
            return 0.0
    return 0.0


def print_optimization_summary(label: str, result) -> None:
    print(f"{label} initial score: {result.initial_eval_score:.3f}")
    print(f"{label} final score:   {result.final_eval_score:.3f}")
    if result.initial_eval_score_per_scorer:
        print(f"{label} initial per scorer: {result.initial_eval_score_per_scorer}")
    if result.final_eval_score_per_scorer:
        print(f"{label} final per scorer:   {result.final_eval_score_per_scorer}")


def should_promote(result) -> bool:
    return (
        result.initial_eval_score is not None
        and result.final_eval_score is not None
        and result.final_eval_score > result.initial_eval_score
    )

# COMMAND ----------

# DBTITLE 1,Notebook Overview
# MAGIC %md
# MAGIC ## Crash Course: PromptOps End to End
# MAGIC
# MAGIC This notebook is a compact PromptOps walkthrough on Databricks:
# MAGIC
# MAGIC 1. Register and version prompts in the MLflow Prompt Registry
# MAGIC 2. Call prompts with a cheap generation model
# MAGIC 3. Evaluate prompt quality with code scorers plus a stronger LLM judge
# MAGIC 4. Optimize a single prompt with MetaPrompt and GEPA
# MAGIC 5. Optimize a small prompt chain and promote only if metrics improve
# MAGIC
# MAGIC **Model strategy**
# MAGIC - **Generation:** `databricks-gpt-oss-20b`
# MAGIC - **Judge / reflection:** `databricks-gpt-5-4`
# MAGIC
# MAGIC We intentionally use an OpenAI GPT endpoint for judge/reflection because the
# MAGIC MLflow judge and MetaPrompt optimization paths expect strict JSON/string outputs.
# MAGIC
# MAGIC **Runtime note**
# MAGIC - Parts 4 and 5 are the longest cells.
# MAGIC - The budgets are intentionally small so this stays a crash course rather than a long benchmark.

# COMMAND ----------

# DBTITLE 1,Part 1 Header
# MAGIC %md
# MAGIC ## Part 1: Basic Prompt Operations
# MAGIC Register, version, alias, and load prompts from the **MLflow Prompt Registry**. Prompts are stored in Unity Catalog with Git-like versioning and mutable aliases for safe deployment.

# COMMAND ----------

# DBTITLE 1,Register Prompts
# Register a simple string prompt (creates v1)
explainer_v1 = mlflow.genai.register_prompt(
    name="demos.prompts.tech_explainer",
    template="Explain {{concept}} in simple terms.",
    commit_message="v1: Basic single-variable explainer",
)
print(f"Registered v1: {explainer_v1.uri}")

# Register a chat-style prompt (creates v2 under the same name)
explainer_v2 = mlflow.genai.register_prompt(
    name="demos.prompts.tech_explainer",
    template=[
        {"role": "system", "content": "You are a technical writer who explains complex topics clearly and concisely for a {{audience}} audience. Keep responses under 100 words."},
        {"role": "user", "content": "Explain {{concept}}."},
    ],
    commit_message="v2: Chat-style with audience targeting and length constraint",
)
print(f"Registered v2: {explainer_v2.uri}")

# COMMAND ----------

# DBTITLE 1,Set Aliases and Load Prompts
# Set deployment aliases
mlflow.genai.set_prompt_alias("demos.prompts.tech_explainer", alias="dev", version=explainer_v2.version)
mlflow.genai.set_prompt_alias("demos.prompts.tech_explainer", alias="prod", version=explainer_v1.version)

# Load by alias - decouple application code from specific versions
dev_prompt = mlflow.genai.load_prompt("prompts:/demos.prompts.tech_explainer@dev")
prod_prompt = mlflow.genai.load_prompt("prompts:/demos.prompts.tech_explainer@prod")

print("Dev prompt (v2 - chat-style):")
for msg in dev_prompt.template:
    print(f"  [{msg['role']}] {msg['content']}")

print(f"\nProd prompt (v1 - simple): {prod_prompt.format(concept='quantum computing')}")

# COMMAND ----------

# DBTITLE 1,Part 2 Header
# MAGIC %md
# MAGIC ## Part 2: Using Prompts with LLMs
# MAGIC Load prompts from the registry and call **Databricks Foundation Model** endpoints. `@mlflow.trace` captures the full execution lineage automatically.

# COMMAND ----------

# DBTITLE 1,Call LLM with Registered Prompt
@mlflow.trace
def explain_concept(concept: str, audience: str = "general") -> str:
    """Load a registered prompt and call the LLM."""
    prompt = mlflow.genai.load_prompt("prompts:/demos.prompts.tech_explainer@dev")
    messages = prompt.format(concept=concept, audience=audience)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=220,
    )
    return extract_text(response.choices[0].message.content)

# Test it
result = explain_concept("transformer architecture in AI", audience="beginner")
print(result)

# COMMAND ----------

# DBTITLE 1,Part 3 Header
# MAGIC %md
# MAGIC ## Part 3: Evaluation with Custom Scorers
# MAGIC Define **code-based** and **LLM-as-a-judge** scorers, then evaluate prompt quality with `mlflow.genai.evaluate()`.

# COMMAND ----------

# DBTITLE 1,Define Custom Scorers
from mlflow.genai.scorers import scorer, Correctness, Guidelines

# ── Code-based scorers ─────────────────────────────────────────────

@scorer
def brevity(outputs: str) -> float:
    """Prefer responses under 100 words. Returns 0-1 score."""
    word_count = len(outputs.split())
    return min(1.0, 100 / max(word_count, 1))

@scorer
def has_key_terms(inputs: dict, outputs: str) -> bool:
    """Check that the response references the concept being explained."""
    concept = inputs.get("concept", "").lower()
    keywords = concept.split()
    return any(kw in outputs.lower() for kw in keywords)

# ── LLM-as-a-judge scorers ────────────────────────────────────────

explanation_quality = Guidelines(
    name="explanation_quality",
    guidelines="The response must explain the concept accurately, clearly, "
               "and at the appropriate level for the target audience. "
               "It should not be overly technical for beginners, oversimplified for experts, "
               "or end abruptly mid-thought.",
    model=JUDGE_MODEL,
)

print("Scorers ready: brevity, has_key_terms, explanation_quality")

# COMMAND ----------

# DBTITLE 1,Build Eval Dataset and Run Evaluation
# Evaluation dataset with inputs and expected responses
eval_records = [
    {
        "inputs": {"concept": "quantum computing", "audience": "beginner"},
        "expectations": {"expected_response": "Quantum computing uses quantum bits (qubits) that can be 0 and 1 simultaneously, enabling faster problem-solving than classical computers."},
    },
    {
        "inputs": {"concept": "blockchain", "audience": "general"},
        "expectations": {"expected_response": "Blockchain is a decentralized digital ledger that records transactions across many computers, making records tamper-resistant."},
    },
    {
        "inputs": {"concept": "neural networks", "audience": "technical"},
        "expectations": {"expected_response": "Neural networks are computational graphs of interconnected nodes organized in layers that learn function approximations through backpropagation."},
    },
    {
        "inputs": {"concept": "API rate limiting", "audience": "developer"},
        "expectations": {"expected_response": "Rate limiting restricts how many API requests a client can make in a time window, protecting servers from overload."},
    },
    {
        "inputs": {"concept": "containerization", "audience": "beginner"},
        "expectations": {"expected_response": "Containerization packages an application with all its dependencies into a lightweight, portable unit that runs consistently anywhere."},
    },
]

eval_data = pd.DataFrame(eval_records)

# Run evaluation across all scorers
eval_results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=explain_concept,
    scorers=[brevity, has_key_terms, explanation_quality, Correctness(model=JUDGE_MODEL)],
)

pd.Series(eval_results.metrics, name="score").sort_index()

# COMMAND ----------

# DBTITLE 1,Part 4 Header
# MAGIC %md
# MAGIC ## Part 4: Single Prompt Optimization
# MAGIC
# MAGIC Two constraints shape the optimization setup:
# MAGIC 1. **Text prompts only** — `optimize_prompts` can only modify string templates, not chat-style message lists. Since `explain_concept` loads the chat-style v2 prompt, we define a text-only optimization prompt here.
# MAGIC 2. **Clear model roles** — the generation model stays cheap (`databricks-gpt-oss-20b`), while a stronger judge (`databricks-gpt-5-4`) scores and reflects.
# MAGIC
# MAGIC Algorithms:
# MAGIC - **MetaPromptOptimizer** – Quick restructuring via prompt-engineering best practices (zero-shot or few-shot)
# MAGIC - **GepaPromptOptimizer** – Iterative refinement using evaluation data and LLM-driven reflection

# COMMAND ----------

# DBTITLE 1,Zero-Shot Optimization (MetaPrompt)
from mlflow.genai.optimize import MetaPromptOptimizer, GepaPromptOptimizer

# Register a text-only prompt whose variables match the optimization dataset.
explainer_opt_seed = mlflow.genai.register_prompt(
    name="demos.prompts.tech_explainer_text",
    template="Explain {{concept}} in simple terms for a {{audience}} audience.",
    commit_message="Text-only seed prompt for optimization demos",
)
print(f"Optimization seed prompt: {explainer_opt_seed.uri}")

# Text-based predict function for optimization.
@mlflow.trace
def explain_for_opt(concept: str, audience: str = "general") -> str:
    prompt = mlflow.genai.load_prompt(explainer_opt_seed.uri)
    text = prompt.format(concept=concept, audience=audience)
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": text}],
        max_tokens=220,
    )
    return extract_text(response.choices[0].message.content)

# Stronger judge for optimization loops
explanation_quality_opt = Guidelines(
    name="explanation_quality",
    guidelines="The response must explain the concept accurately, clearly, "
               "and at the appropriate level for the target audience. "
               "It must finish cleanly without truncation.",
    model=JUDGE_MODEL,
)

# Zero-shot: restructure prompt using best practices (no training data needed)
zero_shot_result = mlflow.genai.optimize_prompts(
    predict_fn=explain_for_opt,
    train_data=[],
    prompt_uris=[explainer_opt_seed.uri],
    optimizer=MetaPromptOptimizer(
        reflection_model=JUDGE_MODEL,
        lm_kwargs={"temperature": 0},
        guidelines="This prompt powers a PromptOps crash-course demo. Improve clarity, "
                   "audience match, and completeness without making responses verbose.",
    ),
    scorers=[],
)

print("── Original Prompt ──")
print(explainer_opt_seed.template)

print("\n── Optimized Prompt (MetaPrompt zero-shot) ──")
print(zero_shot_result.optimized_prompts[0].template)

# COMMAND ----------

# GEPA: iteratively improves using eval feedback and LLM reflection
print(
    f"Starting single-prompt GEPA with {len(eval_records)} samples and "
    f"{SINGLE_PROMPT_MAX_METRIC_CALLS} max metric calls."
)
gepa_start = time.time()
gepa_result = mlflow.genai.optimize_prompts(
    predict_fn=explain_for_opt,
    train_data=eval_records,
    prompt_uris=[explainer_opt_seed.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model=JUDGE_MODEL,
        max_metric_calls=SINGLE_PROMPT_MAX_METRIC_CALLS,
        display_progress_bar=False,
    ),
    scorers=[brevity, has_key_terms, explanation_quality_opt],
)

print(f"Single-prompt GEPA completed in {time.time() - gepa_start:.1f}s")
print_optimization_summary("Single-prompt GEPA", gepa_result)
print("── GEPA-Optimized Prompt ──")
print(gepa_result.optimized_prompts[0].template)

# COMMAND ----------

# DBTITLE 1,Part 5 Header
# MAGIC %md
# MAGIC ## Part 5: Multi-Prompt Optimization
# MAGIC Jointly optimize multiple prompts that work together in a **plan-then-explain** pipeline. A custom weighted objective balances multiple evaluation criteria.

# COMMAND ----------

# DBTITLE 1,Register Multi-Prompt Pipeline
# Register a planning prompt (text template)
plan_prompt = mlflow.genai.register_prompt(
    name="demos.prompts.plan_step",
    template="Create a brief 3-bullet teaching plan for explaining {{concept}} to a {{audience}} audience.",
    commit_message="Initial planning prompt",
)

# Register an explanation prompt (text template — optimization requires text, not chat-style)
explain_prompt = mlflow.genai.register_prompt(
    name="demos.prompts.explain_step",
    template="You are a technical writer. Create a clear explanation for a {{audience}} audience in under 150 words.\n\nConcept: {{concept}}\nPlan:\n{{plan}}",
    commit_message="Initial explain prompt (text format for optimization)",
)

print(f"Plan prompt:    {plan_prompt.uri}")
print(f"Explain prompt: {explain_prompt.uri}")

# COMMAND ----------

# DBTITLE 1,Define Multi-Prompt Pipeline
@mlflow.trace
def plan_and_explain(concept: str, audience: str = "general") -> str:
    """Two-stage pipeline: generate a plan, then explain using that plan."""
    # Stage 1: Generate plan
    plan_tmpl = mlflow.genai.load_prompt(plan_prompt.uri)
    plan_text = plan_tmpl.format(concept=concept, audience=audience)

    plan_response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": plan_text}],
        max_tokens=160,
    )
    plan = extract_text(plan_response.choices[0].message.content)

    # Stage 2: Explain using plan
    explain_tmpl = mlflow.genai.load_prompt(explain_prompt.uri)
    explain_text = explain_tmpl.format(concept=concept, audience=audience, plan=plan)

    explain_response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": explain_text}],
        max_tokens=220,
    )
    return extract_text(explain_response.choices[0].message.content)

# Test the pipeline
result = plan_and_explain("microservices architecture", audience="beginner")
print(result)

# COMMAND ----------

# DBTITLE 1,Multi-Prompt Optimization with Weighted Objective
# Weighted objective balancing multiple criteria
def weighted_objective(scores: dict) -> float:
    return (
        0.45 * score_to_float(scores.get("explanation_quality", 0))
        + 0.35 * score_to_float(scores.get("correctness", 0))
        + 0.10 * score_to_float(scores.get("brevity", 0))
        + 0.10 * score_to_float(scores.get("has_key_terms", 0))
    )

# Jointly optimize both prompts in the pipeline
print(
    f"Starting multi-prompt GEPA with {len(eval_records)} samples and "
    f"{MULTI_PROMPT_MAX_METRIC_CALLS} max metric calls."
)
multi_start = time.time()
multi_result = mlflow.genai.optimize_prompts(
    predict_fn=plan_and_explain,
    train_data=eval_records,
    prompt_uris=[plan_prompt.uri, explain_prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model=JUDGE_MODEL,
        max_metric_calls=MULTI_PROMPT_MAX_METRIC_CALLS,
        display_progress_bar=False,
    ),
    scorers=[brevity, has_key_terms, explanation_quality_opt, Correctness(model=JUDGE_MODEL)],
    aggregation=weighted_objective,
)

print(f"Multi-prompt GEPA completed in {time.time() - multi_start:.1f}s")
print_optimization_summary("Multi-prompt GEPA", multi_result)
# Review optimized prompts
for i, optimized in enumerate(multi_result.optimized_prompts):
    print(f"\n{'='*60}")
    print(f"Optimized Prompt {i+1}")
    print(f"{'='*60}")
    print(optimized.template)

# COMMAND ----------

# DBTITLE 1,Register Optimized Prompts to Prod
# Register optimized prompts as new versions and promote only if the bundle improved.
prompt_names = ["demos.prompts.plan_step", "demos.prompts.explain_step"]
promote_bundle = should_promote(multi_result)

for optimized, name in zip(multi_result.optimized_prompts, prompt_names):
    new_version = mlflow.genai.register_prompt(
        name=name,
        template=optimized.template,
        commit_message="PromptOps crash-course optimization candidate",
    )
    mlflow.genai.set_prompt_alias(name, alias="candidate", version=new_version.version)
    if promote_bundle:
        mlflow.genai.set_prompt_alias(name, alias="prod", version=new_version.version)
        print(f"Promoted {name} v{new_version.version} → @prod")
    else:
        print(
            f"Registered {name} v{new_version.version} as @candidate. "
            "Left @prod unchanged because the optimized bundle did not improve."
        )

# COMMAND ----------

# DBTITLE 1,Wrap-Up
# MAGIC %md
# MAGIC ## What You Just Did
# MAGIC
# MAGIC In one notebook, you walked through the core PromptOps loop:
# MAGIC
# MAGIC 1. **Version prompts** with the MLflow Prompt Registry
# MAGIC 2. **Trace inference** with `@mlflow.trace`
# MAGIC 3. **Score quality** with code-based metrics and a stronger LLM judge
# MAGIC 4. **Optimize prompts** with MetaPrompt and GEPA
# MAGIC 5. **Promote cautiously** by updating aliases only when metrics improve
# MAGIC
# MAGIC This is the shortest useful path from "prompt in a notebook" to a governed PromptOps workflow.