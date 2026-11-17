import sys
from pathlib import Path

# Permite importar os módulos de src/ como no restante do projeto (ex.: `from config import settings`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
