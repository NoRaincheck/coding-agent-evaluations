from __future__ import annotations

from pathlib import Path

from coding_agent_evaluations.runtimes.base import SandboxPaths
from coding_agent_evaluations.runtimes.translate import (
    neutralize_image_only_commands,
    path_mapping,
    rewrite_paths,
    translate_verifier,
)


def make_paths(root: Path) -> SandboxPaths:
    return SandboxPaths(
        root=root,
        workspace=root / "workspace",
        tests=root / "tests",
        logs=root / "logs",
        solution=root / "solution",
        state=root / "state",
    )


def test_longest_path_wins() -> None:
    mapping = {"/logs": "/host/logs", "/logs/verifier": "/host/logs/verifier"}
    text, substitutions = rewrite_paths("cat /logs/verifier/reward.txt", mapping)
    assert text == "cat /host/logs/verifier/reward.txt"
    assert any("/logs/verifier" in item for item in substitutions)


def test_paths_inside_heredocs_are_rewritten() -> None:
    mapping = {"/tests": "/host/tests"}
    text = 'cat > p.py <<EOF\nopen("/tests/config.json")\nEOF\n'
    rewritten, _ = rewrite_paths(text, mapping)
    assert 'open("/host/tests/config.json")' in rewritten


def test_substituted_paths_are_not_rescanned() -> None:
    mapping = {"/tests": "/host/tests", "/host": "/other"}
    text, _ = rewrite_paths("/tests", mapping)
    assert text == "/host/tests"


def test_unrelated_words_are_untouched() -> None:
    mapping = {"/logs": "/host/logs"}
    text, substitutions = rewrite_paths(
        "see /logsfoo /logs_extra https://x/logs /logs", mapping
    )
    assert text == "see /logsfoo /logs_extra https://x/logs /host/logs"
    assert substitutions


def test_path_boundaries() -> None:
    mapping = {"/app": "/host/app"}
    for original, expected in (
        ('cd /app', "cd /host/app"),
        ('cd "/app"', 'cd "/host/app"'),
        ('/app/x.py', "/host/app/x.py"),
        ("myapp", "myapp"),
        ("/application", "/application"),
    ):
        assert rewrite_paths(original, mapping)[0] == expected


def test_image_only_commands_are_neutralized() -> None:
    text, skipped = neutralize_image_only_commands(
        "apt-get update && apt-get install -y curl\n"
        "source /opt/miniconda3/bin/activate\n"
        "conda activate testbed\n"
        "python3 -m pip install -e .\n"
        "pytest /host/tests\n"
    )
    neutralized = [
    line for line in text.splitlines() if line.startswith("# cae:")
]
    assert len(neutralized) == 3
    # The original text is preserved in the comment as an audit trail.
    assert "apt-get install -y curl" in neutralized[0]
    assert any("conda activate testbed" in line for line in neutralized)
    assert not any(
        line.startswith(("apt-get", "source /opt", "conda ")) for line in text.splitlines()
    )
    # `pip install` still works inside the workspace environment.
    assert "python3 -m pip install -e ." in text
    assert "pytest /host/tests" in text
    assert len(skipped) == 3


def test_heredoc_bodies_are_not_neutralized() -> None:
    text, skipped = neutralize_image_only_commands(
        "cat > p.sh <<'EOF'\napt-get install -y curl\nEOF\n"
    )
    assert "apt-get install -y curl" in text
    assert skipped == []


def test_path_mapping_covers_the_benchmark_layouts(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    mapping = path_mapping(paths, {"repo_path": "/testbed"})
    assert mapping["/testbed"] == str(paths.workspace)
    assert mapping["/tests"] == str(paths.tests)
    assert mapping["/logs/verifier"] == str(paths.logs / "verifier")
    # `/tmp` must stay real: the reference leaf runner hardcodes it.
    assert not any(key.startswith("/tmp") for key in mapping)
    # Longest keys first, so no key is a prefix of an earlier one.
    keys = list(mapping)
    assert keys == sorted(keys, key=len, reverse=True)


def test_translate_verifier_reports_an_audit_trail(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    script = tmp_path / "test.sh"
    script.write_text(
        "#!/bin/bash\ncd /testbed\napt-get update\n"
        'echo 1 > /logs/verifier/reward.txt\ncat /tests/config.json\n',
        encoding="utf-8",
    )
    translation = translate_verifier(script, paths, {"repo_path": "/testbed"})
    assert f"cd {paths.workspace}" in translation.text
    assert f"{paths.logs / 'verifier'}/reward.txt" in translation.text
    assert f"{paths.tests}/config.json" in translation.text
    assert "apt-get" not in translation.text.replace(
        "image-only command neutralized: apt-get update", ""
    )
    assert translation.skipped == ["apt-get update"]
    assert translation.as_dict()["skipped_image_only_commands"] == ["apt-get update"]


def test_translate_verifier_is_a_no_op_without_matches(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    script = tmp_path / "test.sh"
    script.write_text("pytest -q tests\n", encoding="utf-8")
    translation = translate_verifier(script, paths, {"repo_path": "/app"})
    assert translation.text == "pytest -q tests\n"
    assert translation.substitutions == []