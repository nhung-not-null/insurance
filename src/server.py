import json

from mcp.server.fastmcp import FastMCP

from src.config import MCP_SERVER_NAME

mcp_server = FastMCP(MCP_SERVER_NAME)


@mcp_server.tool()
def submit_evaluation(score: int, raisonnement: str, est_valide: bool) -> str:
    """Enregistre le score et la validation via le serveur MCP."""
    return json.dumps({"score": score, "raisonnement": raisonnement, "est_valide": est_valide})
