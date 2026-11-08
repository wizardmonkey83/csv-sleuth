import tempfile
import shutil
import subprocess
import sys
import os

from pathlib import Path
from dataclasses import dataclass

@dataclass
class Output:
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    notes: str | None = None


script_dir = Path(__file__).resolve().parent
HARNESS_PATH = script_dir / "harness.py"

TIMEOUT_S = 30.0

def _child_env():
    return {
        "PATH": os.environ.get("PATH", ""),
        "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }

def run_code(csv_data_path: str, code: str, timeout: float = TIMEOUT_S):
    if not csv_data_path:
        raise ValueError("Error: no csv_data_path provided.")

    if not code:
        return Output(notes="No code was provided.")
    
    with tempfile.TemporaryDirectory(prefix="sleuth-") as tmp_dir:
        workdir = Path(tmp_dir)
        file_to_run = workdir / "agent_code.py"

        shutil.copy(Path(csv_data_path), workdir / "data.csv")
        file_to_run.write_text(code)

        try:

            process = subprocess.run(
                [sys.executable, "-I", str(HARNESS_PATH)],
                cwd=workdir,
                env=_child_env(),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S
            )

            return Output(exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr)

        except subprocess.TimeoutExpired as exc:
            return Output(stdout=process.stdout, stderr=process.stderr, notes="Process timed out. Process output until that point was captured and returned.")