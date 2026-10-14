"""Chạy notebook bằng kernel mới, lưu output sau khi tất cả cell thành công."""
from pathlib import Path
import json
import os
import sys

LAB = Path(__file__).resolve().parent
WORKSPACE = LAB.parent
DEPS = WORKSPACE / ".lab01-deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
    os.environ["PYTHONPATH"] = str(DEPS) + os.pathsep + os.environ.get("PYTHONPATH", "")

import nbformat
from nbclient import NotebookClient


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    kernel_root = WORKSPACE / ".lab04-kernels"
    kernel_dir = kernel_root / "kernels" / "lab04"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python LAB 04", "language": "python"}
    (kernel_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(kernel_root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    runtime = WORKSPACE / ".lab04-runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(WORKSPACE / ".lab04_ipython")
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    notebook_path = LAB / "classification.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=1800, kernel_name="lab04",
                            startup_timeout=60,
                            resources={"metadata": {"path": str(LAB)}}, allow_errors=False)
    def on_start(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Đang chạy cell {cell_index}", flush=True)
    client.on_cell_start = on_start
    print("Bắt đầu kernel Jupyter mới cho LAB 04...", flush=True)
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, runtime / "failed_lab04.ipynb")
        raise
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Kernel mới: toàn bộ {count} code cell thành công; đã lưu output.", flush=True)


if __name__ == "__main__":
    main()
