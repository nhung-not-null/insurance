import os
import sys
import json
import argparse
import re
from pathlib import Path
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
import data_designer.config as dd
from data_designer.interface import DataDesigner
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

MODEL_ALIAS = os.getenv("MODEL_ALIAS", "rizlum_slm")
ENDPOINT = os.getenv("ENDPOINT", "http://localhost:6000/v1")
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "insurance-evaluator")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))

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

def load_prompt_content(path_or_str):
    """Charge le contenu d'un fichier si le chemin existe, sinon retourne la chaine brute."""
    if path_or_str and os.path.exists(path_or_str):
        with open(path_or_str, "r", encoding="utf-8") as f:
            return f.read().strip()
    return path_or_str

def get_pipeline_builder(chunks, args):
    """Configure le pipeline pour generer des conversations en injectant les prompts charges."""
    model_cfg = dd.ModelConfig(
        alias=MODEL_ALIAS, model=MODEL_ALIAS, provider="local-provider",
        inference_parameters=dd.ChatCompletionInferenceParams(temperature=0.3)
    )
    tool_configs = []
    if not args.skip_eval:
        tool_configs.append(dd.ToolConfig(
            tool_alias="judge-tools", providers=[MCP_SERVER_NAME],
            allow_tools=["submit_evaluation"], max_tool_call_turns=3
        ))
    
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_cfg], tool_configs=tool_configs)
    
    builder.add_column(dd.SamplerColumnConfig(
        name="context", sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=chunks)
    ))
    
    system_prompt_content = load_prompt_content(args.system_prompt)
    conv_prompt_content = load_prompt_content(args.conv_prompt)
    
    if "{{context}}" not in system_prompt_content:
        raise ValueError("The system prompt must contain the '{{context}}' placeholder.")
    if "{{context}}" not in conv_prompt_content or "{{assistant_system_prompt}}" not in conv_prompt_content:
        raise ValueError("The conversation prompt must contain both '{{context}}' and '{{assistant_system_prompt}}' placeholders.")
    
    builder.add_column(dd.LLMTextColumnConfig(
        name="assistant_system_prompt", model_alias=MODEL_ALIAS,
        prompt=system_prompt_content
    ))
    
    builder.add_column(dd.LLMTextColumnConfig(
        name="conv_json", model_alias=MODEL_ALIAS,
        prompt=conv_prompt_content
    ))
    
    if not args.skip_eval:
        eval_prompt_content = load_prompt_content(args.eval_prompt)
        if "{{conv_json}}" not in eval_prompt_content or "{{context}}" not in eval_prompt_content:
            raise ValueError("The evaluation prompt must contain both '{{conv_json}}' and '{{context}}' placeholders.")
        
        builder.add_column(dd.LLMTextColumnConfig(
            name="evaluation", model_alias=MODEL_ALIAS, tool_alias="judge-tools", with_trace="all_messages",
            prompt=eval_prompt_content
        ))
    return builder

def run_design_process(builder, base_count, skip_eval):
    """Execute la generation de donnees via le DataDesigner."""
    lp = dd.ModelProvider(name="local-provider", endpoint=ENDPOINT, provider_type="openai", api_key="no-key")
    mcp_providers = []
    if not skip_eval:
        mcp_providers.append(dd.LocalStdioMCPProvider(name=MCP_SERVER_NAME, command=sys.executable, args=[str(Path(__file__).resolve()), "serve"]))
    designer = DataDesigner(model_providers=[lp], mcp_providers=mcp_providers)
    return designer.preview(builder, num_records=base_count * NUM_EPOCHS)

def transform_to_sharegpt(df, skip_eval):
    """Convertit le format brut en liste plate de messages avec score conditionnel."""
    final_data = []
    for _, row in df.iterrows():
        try:
            match_conv = re.search(r'(\[.*\])', row['conv_json'], re.DOTALL)
            raw_list = json.loads(match_conv.group(1)) if match_conv else json.loads(row['conv_json'])
            formatted_msgs = [{"role": "system", "content": row['assistant_system_prompt'].strip()}]
            for item in raw_list:
                if "role" in item: formatted_msgs.append(item)
                else: 
                    if "user" in item: formatted_msgs.append({"role": "user", "content": item["user"]})
                    if "assistant" in item: formatted_msgs.append({"role": "assistant", "content": item["assistant"]})
            
            entry = {"messages": formatted_msgs}
            if not skip_eval:
                score = 0
                for msg in row.get('evaluation__trace', []):
                    if msg.get("tool_calls"):
                        args_mcp = json.loads(msg["tool_calls"][0]["function"]["arguments"])
                        score = int(args_mcp.get("score", 0))
                        if score > 0: break
                entry["score"] = score
            
            final_data.append(entry)
        except:
            continue
    return final_data

def main():
    parser = argparse.ArgumentParser(description="Synthetic data generation pipeline with optional LLM-as-a-judge evaluation.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to the input markdown directory.")
    parser.add_argument("--skip-eval", action="store_true", help="Bypass the local MCP server and evaluation phase.")
    
    parser.add_argument("--system-prompt", type=str, default="prompts/system_prompt.txt",
                        help="Path to the TXT file containing the system prompt template (or raw string template).")
    parser.add_argument("--conv-prompt", type=str, default="prompts/conv_prompt.txt",
                        help="Path to the TXT file containing the conversation prompt template (or raw string template).")
    parser.add_argument("--eval-prompt", type=str, default="prompts/eval_prompt.txt",
                        help="Path to the TXT file containing the evaluation prompt template (or raw string template).")
    
    parser.add_subparsers(dest="cmd").add_parser("serve")
    args = parser.parse_args()
    
    if args.cmd == "serve":
        mcp_server.run()
        return

    # Generation automatique des fichiers de prompts par défaut s'ils n'existent pas
    # pour garantir que le script fonctionne "out of the box"
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True)
    
    default_prompts = {
        "system_prompt.txt": "Write a reasonable system prompt in French for an insurance assistant. Input: {{context}}. Use Sujet+Verbe+Complement, active voice, 'nous' only.",
        "conv_prompt.txt": "En te basant sur : {{context}} et les consignes : {{assistant_system_prompt}}\nGenere une conversation de 3 tours entre 'user' et 'assistant' au format liste JSON.\nL'assistant repond uniquement par texte, sans utiliser d'outils.",
        "eval_prompt.txt": "Evalue la conversation {{conv_json}} sur une échelle de 1 a 5.\nVerifie la fidelite au contrat {{context}} et le respect du style 'assistant_system_prompt'.\nUtilise submit_evaluation."
    }
    
    for filename, content in default_prompts.items():
        file_path = prompts_dir / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    try:
        chunks = load_markdown_chunks(args.data_dir)
        builder = get_pipeline_builder(chunks, args)
        results = run_design_process(builder, len(chunks), args.skip_eval)
        
        with open("dataset_final.json", "w", encoding="utf-8") as f:
            json.dump(transform_to_sharegpt(results.dataset, args.skip_eval), f, indent=2, ensure_ascii=False)
    except ValueError as e:
        parser.error(str(e))

if __name__ == "__main__":
    main()