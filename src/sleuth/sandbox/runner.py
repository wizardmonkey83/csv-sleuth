import tempfile
import shutil
import subprocess
import sys
import os
import json

from pathlib import Path
from dataclasses import dataclass
from langsmith import traceable

@dataclass
class Output:
    # running the code
    ok: bool | None = None
    result: str | None = None
    error: str | None = None
    traceback: str | None = None
    timed_out: bool | None = None

    # process 
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    notes: str | None = None


script_dir = Path(__file__).resolve().parent
HARNESS_PATH = script_dir / "harness.py"

TIMEOUT_S = 30.0
TEXT_LIMIT = 15000

def _child_env():
    return {
        "PATH": os.environ.get("PATH", ""),
        "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }

def _truncate(text: str | None, limit: int = TEXT_LIMIT) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _as_text(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value

@traceable
def run_code(code: str, csv_data_path: str, timeout: float = TIMEOUT_S):
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
                timeout=timeout,
            )

        except subprocess.TimeoutExpired as exc:
            return Output(
                ok=False,
                timed_out=True,
                error=f"Timed out after {timeout:g}s.",
                stdout=_truncate(_as_text(exc.stdout)),
                stderr=_truncate(_as_text(exc.stderr)),
            )

        report_path = workdir / "report.json"
        if not report_path.exists():
            return Output(
                ok=False,
                error="Sandbox exited without writing a report (it may have crashed).",
                exit_code=process.returncode,
                stdout=_truncate(process.stdout),
                stderr=_truncate(process.stderr),
            )

        report = json.loads(report_path.read_text(encoding="utf-8"))
        return Output(
            ok=report["ok"],
            result=_truncate(report["result"]),
            error=report["error"],
            traceback=_truncate(report["traceback"]),
            exit_code=process.returncode,
            stdout=_truncate(process.stdout),
            stderr=_truncate(process.stderr),
        )