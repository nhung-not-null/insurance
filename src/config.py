import os
from dotenv import load_dotenv

load_dotenv()

MODEL_ALIAS: str = os.getenv("MODEL_ALIAS", "rizlum_slm")
ENDPOINT: str = os.getenv("ENDPOINT", "http://localhost:6000/v1")
MCP_SERVER_NAME: str = os.getenv("MCP_SERVER_NAME", "insurance-evaluator")
NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "1"))
