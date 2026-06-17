import os
from pathlib import Path


def load_dotenv(dotenv_path=None, *args, **kwargs):
    if dotenv_path is None:
        candidates = [Path.cwd() / ".env", *[parent / ".env" for parent in Path.cwd().parents]]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            return False
    else:
        path = Path(dotenv_path)
        if not path.exists():
            return False

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
            if key == "OPENAI_API_KEY":
                os.environ.setdefault("OPENAI_API_KEY1", value)
            elif key == "OPENAI_API_KEY1":
                os.environ.setdefault("OPENAI_API_KEY", value)
            elif key == "OPENAI_BASE_URL":
                os.environ.setdefault("OLLAMA_BASE_URL", value)
            elif key == "OLLAMA_BASE_URL":
                os.environ.setdefault("OPENAI_BASE_URL", value)
    return True
