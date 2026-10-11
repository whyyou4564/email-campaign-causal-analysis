"""无需本地 kernel 网络通信，逐单元执行并保存教学 Notebook 的真实输出。"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

def main():
    rows = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        namespace = {"__name__": "__notebook__", "__file__": str(path)}
        execution_count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(cell.source, str(path), "exec"), namespace)
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = [
                nbformat.v4.new_output("stream", name="stdout", text=output.getvalue())
            ] if output.getvalue() else []
        notebook.metadata["execution_method"] = "sequential Python cell execution; no Jupyter kernel"
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        rows.append({"notebook": path.name, "code_cells_executed": execution_count, "status": "passed"})
        print(f"PASS {path.name}: {execution_count} code cells")
    (ROOT / "reports/tables/notebook_validation.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
