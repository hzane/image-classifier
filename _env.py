import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).parent


def add_directory(dir: str):
    dir = Path(project_root(), dir).resolve()
    if dir not in sys.path:
        sys.path.insert(0, str(dir))


def append_project_module_path() -> None:
    add_directory('xclas')


# append_project_module_path()
