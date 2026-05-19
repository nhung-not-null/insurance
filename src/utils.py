import os
from pathlib import Path


DEFAULT_PROMPTS: dict[str, str] = {
    "system_prompt.txt": (
        "Write a reasonable system prompt for a helpful AI assistant with expertise in "
        "the insurance sector and coherent with the following input: {{context}}\n"
        "The AI assistant must not engage in harmful behaviors. "
        "The system prompt must be in French and descriptive. "
        "It should follow the following rules:\n"
        "  * Utiliser le pr\u00e9sent de l'indicatif et la voix active.\n"
        "  * Appliquer la structure Sujet + Verbe + Compl\u00e9ment.\n"
        '  * Utiliser exclusivement le \"nous\".\n'
        "  * R\u00e9diger des phrases courtes de maximum 20 mots.\n"
        "  * Pr\u00e9senter les faits de mani\u00e8re directe et p\u00e9dagogique sans donner d'avis.\n"
        "  * Conserver l'int\u00e9gralit\u00e9 des informations relatives aux ann\u00e9es et aux documents mentionn\u00e9s.\n"
        "  * Utiliser un vocabulaire simple, sans jargon ni adjectifs inutiles.\n"
        "Just return the system prompt without any additional text or formatting."
    ),
    "conv_prompt.txt": (
        "Tu es un g\u00e9n\u00e9rateur de datasets de haute qualit\u00e9.\n"
        "En te basant sur ce texte : {{context}}\n"
        "et les consignes du system prompt : {{assistant_system_prompt}}\n\n"
        "G\u00e9n\u00e8re une conversation de exactement {{num_turns}} tours entre un Assur\u00e9 (alias: user) et un Expert (alias: assistant).\n"
        "L'assur\u00e9 doit poser des questions de plus en plus pr\u00e9cises.\n"
        "L'expert r\u00e9pond avec rigueur en citant les termes du contrat.\n"
        "L'assur\u00e9 peut parfois demander des questions qui ne sont pas directement li\u00e9es au contenu ou des connaissances que "
        "l'expert ne connais pas, et l'expert doit refuser d'y r\u00e9pondre pour rester fid\u00e8le \u00e0 sa mission.\n"
        "Format de sortie : JSON avec une liste d'objets contenant 'role' (user ou assistant) et 'content'.\n"
        "Retourne uniquement la liste JSON, sans texte suppl\u00e9mentaire."
    ),
    "eval_prompt.txt": (
        "Evalue la conversation {{conv_json}} sur une échelle de 1 a 5.\n"
        "Verifie la fidelite au contrat {{context}} et le respect du style 'assistant_system_prompt'.\n"
        "Utilise submit_evaluation."
    ),
}


def load_prompt_content(path_or_str: str) -> str:
    """Charge le contenu d'un fichier si le chemin existe, sinon retourne la chaine brute."""
    if path_or_str and os.path.exists(path_or_str):
        with open(path_or_str, "r", encoding="utf-8") as f:
            return f.read().strip()
    return path_or_str


def ensure_default_prompts(prompts_dir: str = "prompts") -> None:
    """Cree les fichiers de prompts par defaut s'ils n'existent pas encore."""
    directory = Path(prompts_dir)
    directory.mkdir(exist_ok=True)
    for filename, content in DEFAULT_PROMPTS.items():
        file_path = directory / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
