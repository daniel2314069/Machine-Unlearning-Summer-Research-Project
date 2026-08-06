import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_has_no_edited_weight_loading_or_oce_call():
    package = ROOT / "concept_clustering"
    violations = []

    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] == "oce":
                        violations.append((path.name, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".", 1)[0] == "oce":
                    violations.append(
                        (path.name, node.lineno, f"from {node.module} import ...")
                    )
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    called_name = node.func.attr
                else:
                    continue
                if called_name in {"load_state_dict", "Orthogonal_Erase", "save_file"}:
                    violations.append((path.name, node.lineno, called_name))

    assert not violations
