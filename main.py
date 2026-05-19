import argparse
import json
from pathlib import Path

from src.chunkers import create_chunk_combinations, load_markdown_chunks, load_markdown_chunks_small
from src.config import NUM_EPOCHS
from src.pipeline import build_pipeline, run_pipeline
from src.server import mcp_server
from src.transforms import transform_to_sharegpt
from src.utils import ensure_default_prompts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic data generation pipeline with optional LLM-as-a-judge evaluation."
    )
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to the input markdown file or directory.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Bypass the local MCP server and evaluation phase.")
    parser.add_argument("--system-prompt", type=str, default="prompts/system_prompt.txt",
                        help="Path to the TXT file containing the system prompt template (or raw string template).")
    parser.add_argument("--conv-prompt", type=str, default="prompts/conv_prompt.txt",
                        help="Path to the TXT file containing the conversation prompt template (or raw string template).")
    parser.add_argument("--eval-prompt", type=str, default="prompts/eval_prompt.txt",
                        help="Path to the TXT file containing the evaluation prompt template (or raw string template).")

    chunk_grp = parser.add_argument_group("chunking")
    chunk_grp.add_argument("--chunk-mode", type=str, choices=["h1", "small", "combinations"], default="h1",
                           help="Chunking strategy: 'h1' splits by H1 (default), 'small' splits by H2/H3, "
                                "'combinations' creates random combos of H2/H3 chunks.")
    chunk_grp.add_argument("--num-combinations", type=int, default=100,
                           help="Number of chunk combinations to generate (used with --chunk-mode=combinations).")
    chunk_grp.add_argument("--combo-sizes", type=str, default="2,3",
                           help="Comma-separated combination sizes (used with --chunk-mode=combinations). Default: '2,3'.")
    chunk_grp.add_argument("--combo-seed", type=int, default=42,
                           help="Random seed for reproducible chunk combinations.")

    turns_grp = parser.add_argument_group("variable turns")
    turns_grp.add_argument("--variable-turns", action="store_true",
                           help="Add a num_turns sampler column for variable-length conversations. "
                                "The conversation prompt must contain the '{{num_turns}}' placeholder.")
    turns_grp.add_argument("--max-turns", type=int, default=19,
                           help="Maximum number of turns when using --variable-turns (default: 19).")

    model_grp = parser.add_argument_group("model config")
    model_grp.add_argument("--temperature", type=float, default=0.3,
                           help="Sampling temperature (default: 0.3).")
    model_grp.add_argument("--top-p", type=float, default=None,
                           help="Top-p nucleus sampling parameter.")
    model_grp.add_argument("--max-tokens", type=int, default=None,
                           help="Maximum tokens per generation.")

    gen_grp = parser.add_argument_group("generation")
    gen_grp.add_argument("--full-generation", action="store_true",
                         help="Use designer.create() for a full persistent dataset instead of preview().")
    gen_grp.add_argument("--num-records", type=int, default=None,
                         help="Override number of records to generate (default: len(chunks) * NUM_EPOCHS).")
    gen_grp.add_argument("--output", type=str, default="dataset_final.json",
                         help="Output JSON file path (default: dataset_final.json).")

    parser.add_subparsers(dest="cmd").add_parser("serve")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cmd == "serve":
        mcp_server.run()
        return

    ensure_default_prompts()

    try:
        if args.chunk_mode == "h1":
            chunks = load_markdown_chunks(args.data_dir)
        elif args.chunk_mode == "small":
            chunks = load_markdown_chunks_small(args.data_dir)
        else:
            small_chunks = load_markdown_chunks_small(args.data_dir)
            combo_sizes = tuple(int(s) for s in args.combo_sizes.split(","))
            chunks = create_chunk_combinations(
                small_chunks,
                num_combinations=args.num_combinations,
                combo_sizes=combo_sizes,
                seed=args.combo_seed,
            )

        builder = build_pipeline(chunks, args)
        num_records = args.num_records if args.num_records is not None else len(chunks) * NUM_EPOCHS
        dataset = run_pipeline(
            builder,
            num_records=num_records,
            skip_eval=args.skip_eval,
            entry_point=str(Path(__file__).resolve()),
            full_generation=args.full_generation,
        )

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(transform_to_sharegpt(dataset, args.skip_eval), f, indent=2, ensure_ascii=False)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()