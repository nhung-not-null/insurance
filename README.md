# Synthetic Data Generation

This project automatically generates a high-quality conversational synthetic dataset ready for fine-tuning (ShareGPT format). It utilizes reference documents (e.g., insurance contracts in Markdown format) and runs on the NVIDIA NeMo Data Designer architecture.

A key feature of this project is the integration of the **Model Context Protocol (MCP)** to implement a "LLM-as-a-Judge". The model automatically evaluates and scores the fidelity of the generated conversations using a local tool (Tool Calling).

## Features

* **Context Extraction:** Automatically splits Markdown files by sections.
* **System Prompt Generation:** Dynamically creates system instructions tailored to the context.
* **Multi-turn Conversations:** Generates consistent exchanges between a `user` and an `assistant`.
* **LLM-as-a-Judge (MCP):** Automated quality evaluation via function calling (Tool Use).
* **ShareGPT Export:** Final formatting in JSON optimized for Supervised Fine-Tuning (SFT).

## Prerequisites and Installation

This project uses `uv` for package management.

1. Clone the repository:
   ```bash
   git clone <your-gitlab-url>
   cd synthetic-data-generation
   ```
2. Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

3. Place your source documents in `.md` format into a `./data` folder at the root of the project.

## Configuration

Create a `.env` file at the root of the project to configure your local LLM environment (e.g., vLLM or Ollama) and generation parameters:

```env
MODEL_ALIAS=rizlum_slm
ENDPOINT=http://localhost:6000/v1
MCP_SERVER_NAME=insurance-evaluator
```


## Usage

The script features a dual-execution system leveraging `argparse` to handle both the NeMo orchestration and the MCP server within a single file.

**Full Generation (with LLM-as-a-Judge evaluation):**

```bash
uv run python main.py
```

**Classic Generation (without evaluation):**
If you want to generate raw text data without spawning the MCP server or running evaluations:

```bash
uv run python main.py --skip-eval
```

## Output Data Structure

The script outputs a `dataset_final.json` file structured in the ShareGPT format:

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "Your generated system prompt..."
      },
      {
        "role": "user",
        "content": "Insured's question..."
      },
      {
        "role": "assistant",
        "content": "Expert's answer..."
      }
    ],
    "score": 5
  }
]
```