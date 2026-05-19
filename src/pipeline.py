import argparse
import sys
from pathlib import Path

import data_designer.config as dd
from data_designer.interface import DataDesigner

from src.config import ENDPOINT, MCP_SERVER_NAME, MODEL_ALIAS
from src.utils import load_prompt_content


def build_pipeline(chunks: list[str], args: argparse.Namespace) -> dd.DataDesignerConfigBuilder:
    """Configure le pipeline pour generer des conversations en injectant les prompts charges."""
    inference_kwargs: dict = {"temperature": args.temperature}
    if args.top_p is not None:
        inference_kwargs["top_p"] = args.top_p
    if args.max_tokens is not None:
        inference_kwargs["max_tokens"] = args.max_tokens

    model_cfg = dd.ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ALIAS,
        provider="local-provider",
        inference_parameters=dd.ChatCompletionInferenceParams(**inference_kwargs),
    )

    tool_configs = []
    if not args.skip_eval:
        tool_configs.append(
            dd.ToolConfig(
                tool_alias="judge-tools",
                providers=[MCP_SERVER_NAME],
                allow_tools=["submit_evaluation"],
                max_tool_call_turns=3,
            )
        )

    builder = dd.DataDesignerConfigBuilder(model_configs=[model_cfg], tool_configs=tool_configs)

    builder.add_column(
        dd.SamplerColumnConfig(
            name="context",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=chunks),
        )
    )

    if args.variable_turns:
        builder.add_column(
            dd.SamplerColumnConfig(
                name="num_turns",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=list(range(1, args.max_turns + 1))),
            )
        )

    system_prompt_content = load_prompt_content(args.system_prompt)
    conv_prompt_content = load_prompt_content(args.conv_prompt)

    if "{{context}}" not in system_prompt_content:
        raise ValueError("The system prompt must contain the '{{context}}' placeholder.")
    if "{{context}}" not in conv_prompt_content or "{{assistant_system_prompt}}" not in conv_prompt_content:
        raise ValueError(
            "The conversation prompt must contain both '{{context}}' and '{{assistant_system_prompt}}' placeholders."
        )
    if args.variable_turns and "{{num_turns}}" not in conv_prompt_content:
        raise ValueError(
            "When using --variable-turns, the conversation prompt must contain the '{{num_turns}}' placeholder."
        )

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="assistant_system_prompt",
            model_alias=MODEL_ALIAS,
            prompt=system_prompt_content,
        )
    )

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="conv_json",
            model_alias=MODEL_ALIAS,
            prompt=conv_prompt_content,
        )
    )

    if not args.skip_eval:
        eval_prompt_content = load_prompt_content(args.eval_prompt)
        if "{{conv_json}}" not in eval_prompt_content or "{{context}}" not in eval_prompt_content:
            raise ValueError(
                "The evaluation prompt must contain both '{{conv_json}}' and '{{context}}' placeholders."
            )
        builder.add_column(
            dd.LLMTextColumnConfig(
                name="evaluation",
                model_alias=MODEL_ALIAS,
                tool_alias="judge-tools",
                with_trace="all_messages",
                prompt=eval_prompt_content,
            )
        )

    return builder


def run_pipeline(
    builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    skip_eval: bool,
    entry_point: str,
    full_generation: bool = False,
):
    """Execute la generation de donnees via le DataDesigner.

    Args:
        builder: configured DataDesignerConfigBuilder
        num_records: number of records to generate
        skip_eval: whether to skip the MCP evaluation phase
        entry_point: absolute path to the main script (used to spawn the MCP server subprocess)
        full_generation: use designer.create() instead of preview() for a full persistent dataset
    """
    lp = dd.ModelProvider(
        name="local-provider",
        endpoint=ENDPOINT,
        provider_type="openai",
        api_key="no-key",
    )

    mcp_providers = []
    if not skip_eval:
        mcp_providers.append(
            dd.LocalStdioMCPProvider(
                name=MCP_SERVER_NAME,
                command=sys.executable,
                args=[entry_point, "serve"],
            )
        )

    designer = DataDesigner(model_providers=[lp], mcp_providers=mcp_providers)

    if full_generation:
        result = designer.create(builder, num_records=num_records, dataset_name="synthetic_conversations")
        dataset = result.load_dataset()
        if hasattr(dataset, "to_pandas"):
            return dataset.to_pandas()
        return dataset

    return designer.preview(builder, num_records=num_records).dataset
