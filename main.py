import os
import sys
import json
import argparse
import re
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import data_designer.config as dd
from data_designer.interface import DataDesigner
from langchain_text_splitters import MarkdownHeaderTextSplitter

MODEL_ALIAS = "rizlum_slm"
ENDPOINT = "http://localhost:6000/v1"
MCP_SERVER_NAME = "insurance-evaluator"

mcp_server = FastMCP(MCP_SERVER_NAME)

@mcp_server.tool()
def submit_evaluation(score: int, raisonnement: str, est_valide: bool) -> str:
    """Enregistre le score et la validation via le serveur MCP."""
    return json.dumps({"score": score, "raisonnement": raisonnement, "est_valide": est_valide})

def load_markdown_chunks(path):
    """Charge les fichiers markdown et les decoupe par titres de niveau 1."""
    input_dir = Path(path)
    file_chunks = []
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1")])
    for md_file in input_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            sections = splitter.split_text(f.read())
            for section in sections:
                header = section.metadata.get("Header 1", "Contrat")
                file_chunks.append(f"Source: {md_file.name} | Section: {header}\n\n{section.page_content}")
    return file_chunks

def get_pipeline_builder(chunks):
    """Configure le pipeline avec des consignes de formatage JSON strictes."""
    model_cfg = dd.ModelConfig(
        alias=MODEL_ALIAS, model=MODEL_ALIAS, provider="local-provider",
        inference_parameters=dd.ChatCompletionInferenceParams(temperature=0.3)
    )
    tool_cfg = dd.ToolConfig(
        tool_alias="judge-tools", providers=[MCP_SERVER_NAME],
        allow_tools=["submit_evaluation"], max_tool_call_turns=3
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_cfg], tool_configs=[tool_cfg])
    builder.add_column(dd.SamplerColumnConfig(
        name="context", sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=chunks)
    ))
    builder.add_column(dd.LLMTextColumnConfig(
        name="assistant_system_prompt", model_alias=MODEL_ALIAS,
        prompt=(
            "Write a reasonable system prompt for a helpful AI assistant with expertise in "
            "the insurance sector and coherent with the following input: {{context}}\n"
            "The system prompt must be in French and follow these rules:\n"
            "* Present de l'indicatif, voix active, structure Sujet+Verbe+Complement.\n"
            "* Utiliser exclusivement le 'nous', phrases courtes (max 20 mots).\n"
            "* Direct et pedagogique, sans avis, conserver dates et documents.\n"
            "Return only the prompt text."
        )
    ))
    builder.add_column(dd.LLMTextColumnConfig(
        name="conv_json", model_alias=MODEL_ALIAS,
        prompt=(
            "En te basant sur : {{context}} et les consignes : {{assistant_system_prompt}}\n"
            "Genere une liste JSON de messages avec 'role' et 'content'.\n"
            "Format: [{\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]"
        )
    ))
    builder.add_column(dd.LLMTextColumnConfig(
        name="evaluation", 
        model_alias=MODEL_ALIAS, 
        tool_alias="judge-tools", 
        with_trace="all_messages",
        prompt=(
            "Analyse la fidelite de la conversation {{conv_json}} par rapport au contrat {{context}}.\n"
            "Donne une note sur 5 en respectant strictement ce bareme :\n"
            "1 : Contresens majeur ou invention totale.\n"
            "2 : Erreur sur les delais ou les montants.\n"
            "3 : Correct mais manque de precision sur les documents.\n"
            "4 : Tres fidele, une petite maladresse de style.\n"
            "5 : Parfaitement fidele et respecte les consignes de l'assistant.\n\n"
            "Utilise l'outil submit_evaluation pour rendre ton verdict."
        )
    ))
    return builder

def run_design_process(builder, base_count):
    """Execute la generation de donnees via le DataDesigner."""
    lp = dd.ModelProvider(name="local-provider", endpoint=ENDPOINT, provider_type="openai", api_key="no-key")
    mp = dd.LocalStdioMCPProvider(name=MCP_SERVER_NAME, command=sys.executable, args=[str(Path(__file__).resolve()), "serve"])
    designer = DataDesigner(model_providers=[lp], mcp_providers=[mp])
    return designer.preview(builder, num_records=base_count)

def transform_to_sharegpt(df):
    """Extrait les messages et le score de maniere robuste via tool_calls ou regex textuelle."""
    final_data = []
    for _, row in df.iterrows():
        try:
            match_conv = re.search(r'(\[.*\])', row['conv_json'], re.DOTALL)
            messages = json.loads(match_conv.group(1)) if match_conv else json.loads(row['conv_json'])
            messages.insert(0, {"role": "system", "content": row['assistant_system_prompt'].strip()})
            score = 0
            for msg in row.get('evaluation__trace', []):
                if msg.get("tool_calls"):
                    args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
                    score = int(args.get("score", 0))
                    if score > 0: break
            if score == 0 and "evaluation" in row:
                match_score = re.search(r'"score":\s*(\d+)', str(row["evaluation"]))
                if match_score: score = int(match_score.group(1))
            final_data.append({"messages": messages, "score": score})
        except:
            continue
    return final_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_subparsers(dest="cmd").add_parser("serve")
    parsed = parser.parse_args()
    if parsed.cmd == "serve":
        mcp_server.run()
    else:
        chunks = load_markdown_chunks("./data")
        results = run_design_process(get_pipeline_builder(chunks), len(chunks))
        final_json = transform_to_sharegpt(results.dataset)
        with open("dataset_final.json", "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()