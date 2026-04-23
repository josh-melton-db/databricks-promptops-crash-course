# It's Just Text

*A field guide to PromptOps, and why MLflow (not Git) is the right system of record for the loop.*

The AI industry keeps renaming the work.

Prompt engineering, then RAG, then context engineering, then MCP, then "agent skills" and `AGENTS.md` and `CLAUDE.md` and `.cursorrules`, then DSPy and GEPA and `optimize_anything` and Autoresearch, and now harness engineering. Each wave gets more hype than the last.

But the substrate never changes. The model still consumes tokens. Everything new is just a different way to *create* and *route* the text those tokens encode.

A prompt is text. A retrieved chunk is text. A memory summary is text. An MCP tool definition and the JSON it returns are text. A `SKILL.md`, `AGENTS.md`, `SOUL.md`, `.cursorrules`, `CLAUDE.md`, or any other instruction file is text. Source code, diffs, test logs, CLI output, plan steps, judge rubrics — all text.

That single observation is what makes PromptOps a durable discipline rather than a passing label. If your model behavior is determined by text — and it is — then that text needs to be versioned, traced, evaluated, optimized, promoted, and audited with the same care you'd apply to production code. The text artifacts are the thing.

This post walks the eras, shows where this repo's PromptOps notebook fits, and argues that for the loop around the text, MLflow does what Git cannot.

## The eras (overlapping, not sequential)

These are layers, not eras-replacing-eras. RAG and prompt engineering live happily inside MCP-using harnesses today. But each wave added a new kind of text to the agent's input surface and a new way to assemble it.

### 1. Prompt engineering — including RAG and memory

Someone writes a string or chat template and tries to get better behavior. Then they add retrieval — fetch the most relevant chunks from a vector store and inject them into the prompt. Then they add memory — summarize the last N turns, or the last day's interactions, and inject that too.

RAG and memory are sometimes treated as separate disciplines, but they're really the same idea: assemble *more text* into the request. A vector index is a typed lookup over text. A "memory" is just a stored text artifact retrieved into context on demand. Vectors are the index; text is the payload.

This repo starts here. The notebook registers a one-line template — `Explain {{concept}} in simple terms.` — and calls a Foundation Model with it.

### 2. MCP, CLIs, and code — text wrappers over the rest of the world

MCP is the cleanest current proof of the substrate-is-text idea. An MCP server is essentially an API with an agent-friendly text interface bolted on top. The agent never sees HTTP routes or RPC stubs. It sees text descriptions of tools, text resource templates, and text responses returned from tool calls.

CLIs work the same way. The command is text, stdout/stderr/exit codes are text, `--help` and man pages are text. The agent doesn't have a privileged channel into the OS — it has a text protocol.

Source code is the same again. When an agent reads a file or writes a patch, it's operating on text: function signatures, diffs, test output, build logs.

What MCP standardizes is not new capability — it's the *text shape* of the wrapper.

### 3. The markdown era — instructions, skills, configuration as text

This is the wave most teams are riding right now: `SKILL.md`, `AGENTS.md`, `SOUL.md`, `.cursorrules`, `CLAUDE.md`, `BOOTSTRAP.md`, custom instruction packs.

Instead of one prompt, you author a *portfolio* of structured markdown that the agent treats as guidance — loaded conditionally, layered, sometimes mutated by the agent itself. Anthropic's `SKILL.md` standard, Cursor rules, the `AGENTS.md` convention, OpenAI's instruction files, and the long tail of tool-specific configuration files are all variants of the same idea: the model's behavior is configured by markdown the model reads.

Alexander Krentsel's [OpenClaw deep-dive](https://docs.google.com/presentation/d/1vO8GHrJTJGBHO3qc2OTkuQcNx110f1t5juMbe9XVPaQ/edit) makes this very explicit. In OpenClaw, agent identity, persona, tooling preferences, and even safety rules live in a stack of `.md` files (`IDENTITY.md`, `USER.md`, `SOUL.md`, `AGENTS.md`, `TOOLS.md`, `BOOTSTRAP.md`). Krentsel's observation: almost all of the "magic" personalization comes from a few of these files plus a memory summary of recent work. Memory in OpenClaw is a vector DB over past conversations and documents — text that gets retrieved back into context on demand.

The "magic," in other words, isn't a model upgrade. It's text the user (or the agent itself) wrote.

### 4. Harness engineering — the synthesis

Harness engineering is what happens when you put all of the above under a single configurable runtime. Memory + tools (built-in, MCP, CLI) + skills/instruction markdown + an orchestration loop, all configurable, all introspectable. Claude Code, OpenClaw, NanoClaw, PyClaw, Cursor's agent mode, and the rapidly-growing pile of open agentic-coding harnesses all sit here.

The OpenClaw deck frames the field's progression cleanly:

> Phase 0 (2018–2020): LLMs as next-token predictors
> Phase 1 (2022–2024): Fine-tuned LLM as assistants
> Phase 2 (2024–2025): LLM + tool-use as "scoped" agents
> Phase 3 (2025–2026): LLM + tool-use + dynamic tool discovery → autonomous agents
> Phase 4 (2026+): self-evolving systems

And the throughline, in Krentsel's own words: *"all systems boil down to LLM calls. The difference is the context provided."*

That is exactly the thesis. The eras differ in how the context is created. The substrate is constant.

The reason OpenClaw and Claude Code feel like more than a folder of `.md` files is that the runtime around the text *operates on* the text — composes it, traces it, refines it, version-controls it, swaps it out. That operating loop is what we'll come back to in the MLflow section. Without it, you have a prompt soup. With it, you have a harness.

### 5. The optimization-and-autoresearch arc — when the model writes the text

Running through every era is a parallel trend: *automating the production of text by the model itself.*

- **DSPy** (Stanford NLP, 2023) treated prompts as compileable artifacts. You write a program; an optimizer searches for the prompt strings.
- **GEPA -> `optimize_anything*`* is the next step in that same line of thought. GEPA introduced the reflective, eval-driven loop: use judge feedback to iteratively rewrite prompts. `optimize_anything` (GEPA, Feb 2026) is the explicit generalization of that idea to any text artifact: prompts, tool descriptions, planner instructions, code, agent architectures, scheduling policies, even SVG graphics and CUDA kernels. That's why GEPA still shows up in this repo's notebook: `GepaPromptOptimizer` is the prompt-scoped version of the same pattern. The API tagline is, almost word-for-word, the thesis of this post: *"if something can be serialized to a string and its quality measured, an LLM can reason about it and propose improvements."* That is the substrate-is-text claim, in the form of a Python function signature.
- **Autoresearch** — OpenAI Deep Research, Gemini Deep Research, Perplexity Deep Research, Ai2 ScholarQA, Stanford STORM, LangChain Open Deep Research — is the most vogue version right now. An agent autonomously plans, retrieves, synthesizes, and writes a long-form text artifact. The output is text, the control is text, the retrieved sources are text. The model is just doing more of the writing than ever before.

The shape is the same in every case. We've moved from *"human writes text"* to *"human writes a program that writes the text"* to *"human writes a scoring function and the model writes everything else."* Powerful, vogue, and announced as the Next Thing every few months.

But notice what is almost never in those announcements:

- Which version of the optimized prompt is in production right now?
- When the deep-research agent regresses, which of its internal prompts — planner, query rewriter, source ranker, synthesizer — changed last, and which evaluation would have caught it?
- Did the latest `optimize_anything` pass actually score better on a domain-specific scorer, or did it just look better in the demo?

If the answer is "we don't know," you don't have a system. You have a screenshot.

This is the recurring pattern. Every new technique for letting the model produce more of its own text generates immediate excitement, and the operational story catches up later, painfully. PromptOps is the boring layer the buzzwords keep needing.

## What this repo's notebook actually does

`notebooks/promptops_end_to_end.py` is small enough to walk in one read, and it touches every part of that loop.

Register a versioned prompt and assign deployment aliases — the `commit_message`, the URI, and the `dev`/`prod`/`candidate` aliases are how PromptOps gets *operational* rather than just authorial:

```112:135:notebooks/promptops_end_to_end.py
explainer_v1 = mlflow.genai.register_prompt(
    name="demos.prompts.tech_explainer",
    template="Explain {{concept}} in simple terms.",
    commit_message="v1: Basic single-variable explainer",
)

explainer_v2 = mlflow.genai.register_prompt(
    name="demos.prompts.tech_explainer",
    template=[
        {"role": "system", "content": "You are a technical writer ... for a {{audience}} audience. Keep responses under 100 words."},
        {"role": "user", "content": "Explain {{concept}}."},
    ],
    commit_message="v2: Chat-style with audience targeting and length constraint",
)

mlflow.genai.set_prompt_alias("demos.prompts.tech_explainer", alias="dev", version=explainer_v2.version)
mlflow.genai.set_prompt_alias("demos.prompts.tech_explainer", alias="prod", version=explainer_v1.version)
```

Trace a call so the runtime behavior is captured against a specific prompt version:

```157:169:notebooks/promptops_end_to_end.py
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
```

Score it with code metrics, an LLM judge, and a domain-specific guideline scorer:

```245:249:notebooks/promptops_end_to_end.py
eval_results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=explain_concept,
    scorers=[brevity, has_key_terms, explanation_quality, Correctness(model=JUDGE_MODEL)],
)
```

Then close the loop: feed the *same scoring functions* back into a prompt optimizer that rewrites the text using the eval signal it just produced.

```423:434:notebooks/promptops_end_to_end.py
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
```

And only promote when the metrics actually improve:

```69:74:notebooks/promptops_end_to_end.py
def should_promote(result) -> bool:
    return (
        result.initial_eval_score is not None
        and result.final_eval_score is not None
        and result.final_eval_score > result.initial_eval_score
    )
```

That's a small notebook, but it has the entire operational shape of harness engineering: text artifact → trace → evaluation → optimizer → promotion gate.

## The triad that earns the word "harness"

Markdown alone doesn't get you a harness. Three capabilities are what turn "I have a folder of `.md` files" into "I have a system I can improve."

**Tracing is the runtime ground truth.** `@mlflow.trace` records what text actually ran, against which version, with which intermediate steps, and with which output. Without it, you can't connect a behavior to a cause. With it, every prompt version, every tool call, every plan step in `plan_and_explain` is traceable to a specific change in a specific file at a specific time. Tracing is what makes the difference between debugging a system and guessing at it.

**Evaluation is the quality signal.** Generic "helpfulness" doesn't tell you whether *your* harness is doing *your* job. You need scorers tied to your domain — did it follow the policy, escalate at low confidence, choose the right tool, preserve the required legal language, extract the right entity, use the right terminology for this audience. The notebook's `explanation_quality` `Guidelines` scorer is the simplest possible version of that pattern: a rubric written in your domain's language, judged by an LLM, scoped to your task. Without custom evaluation, optimization becomes aesthetic — you're arguing about wording. With it, you can measure whether a new text artifact is actually better for the work you actually do.

**Optimization is what closes the loop.** GEPA and MetaPrompt aren't just "rewrite my prompt." They take the *exact same scorers* you used in evaluation and use them as the search signal for rewriting the text. The eval signal becomes the optimizer's gradient. Look back at the `multi_result` snippet: the same `brevity`, `has_key_terms`, `explanation_quality_opt`, and `Correctness` scorers feed both `evaluate()` and `optimize_prompts()`. Eval and optimization are not two ideas — they're one feedback loop, and the artifact in the middle is text.

These three capabilities only deliver value if they live next to the text they're describing, the dataset they're scored on, and the runs that produced them. Otherwise you're stitching together URLs and CSVs by hand.

## Why MLflow, and why Git is not enough

Git is great at versioning text as source. This repo is in Git, the notebook is in Git, that's all fine.

But Git only answers source-code questions: which line changed, who changed it, when. The questions PromptOps actually has to answer are different:

- Which prompt version produced this trace?
- What did the domain-specific scorer say about that response?
- What did GEPA produce when it ran against this dataset, this scorer set, this aggregation?
- Did the multi-prompt bundle clear the promotion gate?
- Which alias is `prod` pointing at right now, and which evaluation run justifies that?

You can build glue around Git to answer all of those. That is the point. You have to *build the glue.* And the glue is most of the system.

The notebook describes prompt storage as "Git-like versioning" in Unity Catalog. The "Git-like" part is the analogy. The `mlflow.genai.`* APIs around it are what's actually doing the work. A registered prompt is connected, by URI, to the trace that used it, the evaluation run that scored it, the optimizer run that produced its successor, and the alias that decides whether it ships. `should_promote` reads the optimizer's `initial_eval_score` and `final_eval_score` directly — there is no extracting it from a JSON blob in a side repo.

That connectedness is the whole point. The lifecycle of a text artifact in a harness is not a sequence of source changes. It's a graph of versions, traces, datasets, scorer outputs, and promotion decisions. Git treats the text as the only first-class object. MLflow treats all of those as first-class objects, *and links them.*

There's a deeper reason for this, and it's worth saying plainly: **PromptOps has far more in common with MLOps than with traditional software engineering.** Code is reviewed via diffs and tested with deterministic unit tests. Models — and prompts — are evaluated against held-out datasets with metrics, compared across runs, registered with versions, promoted by aliases, and rolled back when production behavior regresses. We don't put model weights in Git, because Git can't reason about a `.safetensors` file the way a registry, an experiment, and an evaluation suite can. Prompts are the same kind of artifact: they're behavior-shaping parameters whose quality is only knowable through evaluation. The notebook makes this concrete — `register_prompt` returns a versioned URI, `evaluate()` produces metrics, `optimize_prompts()` searches for better parameters, and `set_prompt_alias("prod", ...)` is a deployment, not a commit. Treating prompts as code is a category error. They're models, and they belong in the place models live.

A reasonable counter: "Why not just keep prompts as YAML in Git, run evaluations as a CI job, and store results in a database?" You can absolutely do that. You'll spend most of your engineering time building the connections between prompt versions, traces, evaluation runs, optimizer outputs, and aliases. That's a system. MLflow is that system, already built, with native APIs for each piece.

The honest framing is "Git for source, MLflow for the text lifecycle." For everything beyond a single prompt-as-string, MLflow is the right system of record.

## A note on the harness era getting weirder

Krentsel's slides have one line that should haunt anyone shipping an agentic system: *"the agent is becoming the interface for configuring itself."*

If you take that seriously, the operational question becomes: who governs the text the agent is generating about itself? If your harness can rewrite its own `SKILL.md`, update its own memory summary, or revise its own system prompt — and modern harnesses can — you need version control, traces, evaluations, and rollback aliases for *that* text the same way you need them for application prompts. Otherwise the agent's self-improvement is unobservable and ungovernable.

This is exactly the regime MLflow's prompt registry, tracing, evaluation, and optimizer outputs were designed for. As the harness era leans further into self-configuration, the operational loop matters more, not less.

## Try it

If you want to feel the loop for yourself, the smallest useful exercise is:

1. Open `notebooks/promptops_end_to_end.py` and run it top to bottom on Databricks serverless.
2. Pick one new criterion you care about — accuracy on a vocabulary term, refusal on a forbidden topic, a formatting requirement, a tone constraint — and add it as a custom `@scorer` or a new `Guidelines` rubric next to `explanation_quality`.
3. Re-run the multi-prompt GEPA optimization with your new scorer in the list.
4. Watch `should_promote` — does your scorer change which version ships? Does the trace explain why?

That's the whole loop in one sitting: text artifact → trace → domain-specific eval → eval-driven optimization → promotion gate.

Whatever you call the era — prompt engineering, MCP, the markdown era, harness engineering — the work is the same. You're managing text. You may as well manage it well.