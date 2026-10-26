from pathlib import Path


def get_package_dir(package_name='sparq', max_depth=5) -> Path | None:
    current_dir = Path(__file__).parent
    depth = 0
    while depth < max_depth:
        if current_dir.name == package_name:
            return current_dir
        else:
            current_dir = current_dir.parent
            depth += 1
    return None

def get_project_root(max_depth=5) -> Path | None:
    """ Return the package directory"""

    # Get pointer to current directory
    package_dir = Path(__file__).parent
    depth = 0
    while depth < max_depth:
        if (package_dir / "pyproject.toml").exists():
            return package_dir
        else:
            package_dir = package_dir.parent
            depth += 1
    return None