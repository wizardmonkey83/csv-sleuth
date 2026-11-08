import json
import traceback
import socket
import ssl
import pandas as pd

from pathlib import Path

def _no_network(*args, **kwargs):
    raise PermissionError("Network access is disabled in the sandbox.")
 
 
socket.socket = _no_network 
socket.create_connection = _no_network
 
 
def _describe(value):
    if value is None:
        return None
    
    if isinstance(value, pd.DataFrame):
        return value.to_string(max_rows=50, max_cols=20)
    
    if isinstance(value, pd.Series):
        return value.to_string(max_rows=50)
    
    return repr(value)
 
 
report = {
    "ok": True, 
    "result": None, 
    "error": None, 
    "traceback": None
}

try:
    namespace = {"pd": pd, "df": pd.read_csv("data.csv")}
    code = Path("agent_code.py").read_text()
    exec(compile(code, "agent_code.py", "exec"), namespace)
    report["result"] = _describe(namespace.get("result"))

except BaseException as exc: 
    report["ok"] = False
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()

# saves to working dir defined in the subprocess creation
Path("report.json").write_text(json.dumps(report))