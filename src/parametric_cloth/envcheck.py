"""Environment capability check: which optional dependencies are installed,
and therefore which modules of the framework are usable right now.

    parametric-cloth-doctor
    # or: python -m parametric_cloth.envcheck
"""

from __future__ import annotations

import importlib.util
import shutil

# (import name, pip install name, what it unlocks)
_CHECKS = [
    ("smplx", "smplx", "Module 2: SMPL-X avatar generation (also needs licensed model weights)"),
    ("torch", "torch", "Module 2 (smplx), Module 6 (training), Module 9 (FabricPredictor)"),
    ("trimesh", "trimesh", "Module 2: avatar export to OBJ/PLY/GLB"),
    ("onnx", "onnx", "Module 6: export the trained deformation model to ONNX"),
    ("onnxruntime", "onnxruntime", "Module 6/7: ONNX inference (ONNXDeformer)"),
    ("sentence_transformers", "sentence-transformers", "Module 9: FabricPredictor text embeddings"),
    ("matplotlib", "matplotlib", "parametric_cloth.viz plotting helpers"),
    ("pytest", "pytest", "running the test suite"),
]


def _bpy_available() -> bool:
    try:
        import bpy  # noqa: F401
        return True
    except ImportError:
        return False


def check_environment(*, verbose: bool = True) -> dict:
    """Probe optional dependencies and Blender availability.

    Returns a dict of ``{name: bool}`` (import names plus ``"blender"``);
    prints a human-readable report when ``verbose`` (the default).
    """
    results: dict[str, bool] = {}
    for module_name, pip_name, used_by in _CHECKS:
        available = importlib.util.find_spec(module_name) is not None
        results[module_name] = available
        if verbose:
            marker = "[x]" if available else "[ ]"
            status = "OK" if available else "missing"
            line = f"{marker} {module_name:22} {status:8} -> {used_by}"
            if not available:
                line += f"  (pip install {pip_name})"
            print(line)

    blender_found = shutil.which("blender") is not None
    results["blender"] = blender_found
    if verbose:
        marker = "[x]" if blender_found else "[ ]"
        status = "found on PATH" if blender_found else "not found"
        print(f"{marker} {'blender':22} {status:8} -> Modules 3-4: cloth simulation, post-processing")
        if not blender_found:
            print("    (install Blender 3.6+ so the 'blender' command is on PATH)")

        print()
        print("Always usable (pure stdlib, no extras): Module 1 (patterns), the")
        print("Module 8 GarmentCode adapter + rule-based refine, Module 9 fuzzy")
        print("fabric matching.")
        print("Usable with just numpy (already a core dependency): Module 2/3/4/5")
        print("geometry (placement, tessellation, seams, PCA), Module 6 NumpyMLP")
        print("inference, Module 7 (minus ONNXDeformer), Module 10 differentiable")
        print("fitting, the Blender-free preview in parametric_cloth.preview.")

    return results


def main() -> int:
    check_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
