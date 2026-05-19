import json
import re

import pandas as pd


def transform_to_sharegpt(df: pd.DataFrame, skip_eval: bool) -> list[dict]:
    """Convertit le format brut en liste plate de messages avec score conditionnel."""
    final_data = []
    for _, row in df.iterrows():
        try:
            match_conv = re.search(r"(\[.*\])", row["conv_json"], re.DOTALL)
            raw_list = json.loads(match_conv.group(1)) if match_conv else json.loads(row["conv_json"])

            formatted_msgs = [{"role": "system", "content": row["assistant_system_prompt"].strip()}]
            for item in raw_list:
                if "role" in item:
                    formatted_msgs.append(item)
                else:
                    if "user" in item:
                        formatted_msgs.append({"role": "user", "content": item["user"]})
                    if "assistant" in item:
                        formatted_msgs.append({"role": "assistant", "content": item["assistant"]})

            entry: dict = {"context": row["context"], "messages": formatted_msgs}

            if not skip_eval:
                score = 0
                for msg in row.get("evaluation__trace", []):
                    if msg.get("tool_calls"):
                        args_mcp = json.loads(msg["tool_calls"][0]["function"]["arguments"])
                        score = int(args_mcp.get("score", 0))
                        if score > 0:
                            break
                entry["score"] = score

            final_data.append(entry)
        except Exception:
            continue

    return final_data
