# Databricks PromptOps Crash Course

This repository packages the `PromptOps End to End` Databricks notebook as a small, shareable project.

The notebook walks through:

- registering and versioning prompts in the MLflow Prompt Registry
- calling Databricks Foundation Model endpoints from prompt templates
- evaluating responses with code scorers and LLM-as-a-judge scorers
- optimizing prompts with MetaPrompt and GEPA
- promoting prompt candidates only when evaluation improves

## Repository Layout

```text
.
├── notebooks/
│   └── promptops_end_to_end.py
├── requirements.txt
└── README.md
```

## Notebook Source

The notebook in `notebooks/promptops_end_to_end.py` was exported from:

`/Workspace/Users/josh.melton@databricks.com/PromptOps End to End`

## Prerequisites

- A Databricks workspace with Unity Catalog enabled
- Access to MLflow Prompt Registry and GenAI APIs
- Access to the following Databricks-hosted models:
  - `databricks-gpt-oss-20b`
  - `databricks-gpt-5-4`

## Runtime Notes

- The notebook was tuned to keep optimization budgets demo-friendly.
- The recommended serverless runtime for this notebook is serverless environment version `4`.
- Parts 4 and 5 are the longest-running sections because they perform prompt optimization loops.

## Dependencies

Install the notebook dependencies locally or in Databricks using `requirements.txt`, or rely on the notebook's `%pip` setup cell.

## Running In Databricks

1. Import or sync `notebooks/promptops_end_to_end.py` into your Databricks workspace.
2. Attach the notebook to serverless compute.
3. If available, select serverless environment version `4`.
4. Run the notebook from top to bottom.

## Notes

- The notebook uses a cheaper generation model and a stronger judge/reflection model to keep the demo practical.
- Judge/reflection paths are intentionally GPT-based because strict JSON/string outputs were more reliable for optimization workflows.
