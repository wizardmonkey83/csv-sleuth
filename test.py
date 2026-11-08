import pytest

from src.sleuth.sandbox.runner import run_code


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("region,revenue\nEast,100\nWest,200\nEast,50\n")
    return path


def test_returns_result(csv):
    r = run_code("result = df.groupby('region')['revenue'].sum()", csv)
    assert r.ok
    assert "East" in r.result and "150" in r.result


def test_captures_print(csv):
    r = run_code("print(len(df))", csv)
    assert r.ok
    assert r.stdout.strip() == "3"


def test_reports_errors(csv):
    r = run_code("df['missing']", csv)
    assert not r.ok
    assert "KeyError" in r.error
    assert "agent_code.py" in r.traceback


def test_times_out(csv):
    r = run_code("while True:\n    pass", csv, timeout=2)
    assert r.timed_out and not r.ok


def test_hides_api_keys(csv, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    r = run_code("import os\nresult = os.environ.get('OPENAI_API_KEY')", csv)
    assert r.ok
    assert r.result is None


def test_blocks_network(csv):
    r = run_code("import urllib.request\nurllib.request.urlopen('https://example.com')", csv)
    assert not r.ok


def test_original_file_untouched(csv):
    before = csv.read_text()
    r = run_code("df.iloc[0:0].to_csv('data.csv', index=False)", csv)
    assert r.ok
    assert csv.read_text() == before