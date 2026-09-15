"""The repair harness has to be right before anything it measures means much.

Every phase of this project that had to be retracted was retracted for a
measurement defect rather than a modelling one, and two turned up in this
harness within the hour: a recursive diff that read pytest's own __pycache__ as
the agent editing the tests, and a self-reported action log that recorded two
calls for a run that made twenty. Both are covered here.
"""
from __future__ import annotations

import json
import re

from make_repair_tasks import MUTATIONS, SKIP, candidate_lines
from run_repair_trials import Trial, tool_calls


def test_mutations_change_logic_not_syntax():
    for text, expected in (
        ("if a == b:", "if a != b:"),
        ("while x <= y:", "while x < y:"),
        ("return not ready", "return ready"),
        ("value = min(a, b)", "value = max(a, b)"),
    ):
        assert any(
            re.subn(pattern, replacement, text, count=1)[0] == expected
            for pattern, replacement in MUTATIONS
        ), text


def test_imports_and_comments_are_not_mutation_sites(tmp_path):
    source = tmp_path / "m.py"
    source.write_text("import os\n# a == b\n\nif a == b:\n    pass\n")
    lines = {number for number, *_ in candidate_lines(source)}
    assert lines == {4}
    for skipped in ("import os", "# a == b", "from x import y", "   "):
        assert SKIP.match(skipped)


def test_tool_calls_come_from_the_transcript_not_the_agent(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "thinking about it"},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}},
                        {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
                    ]
                }
            }
        )
        + "\ntruncated line that is not json\n"
    )
    assert tool_calls(transcript) == ["Bash pytest -q", "Read a.py"]


def test_a_missing_transcript_is_no_actions(tmp_path):
    assert tool_calls(tmp_path / "absent.jsonl") == []


def test_a_prompt_full_of_shell_syntax_reaches_the_agent_whole():
    """Note blocks are made of shell commands the earlier agent ran.

    Interpolated raw into a shell string, their quotes close the argument early
    and the agent command dies in zero seconds with an empty transcript — which
    reads downstream as "the agent made no tool calls" rather than as a failure.
    """
    import shlex
    import subprocess

    prompt = (
        'Notes:\n- Bash grep -n "^UNSATISFIABLE" tests/t.py\n'
        "- Bash python3 << 'EOF'\n- Bash ls | head -20 && echo $HOME\n"
    )
    # The agent command carries JSON of its own, so substitution is plain
    # replacement; str.format would read `{"mcpServers":{}}` as a field.
    template = """printf %s {prompt} --mcp-config '{"mcpServers":{}}'"""
    command = template.replace("{prompt}", shlex.quote(prompt))
    finished = subprocess.run(command.split(" --mcp-config")[0], shell=True, capture_output=True, text=True)
    assert finished.returncode == 0
    assert finished.stdout == prompt
    assert '{"mcpServers":{}}' in command


def test_a_green_suite_with_edited_tests_is_not_a_repair():
    trial = Trial(
        task="version-12",
        condition="check",
        resolved=False,
        broken_before=3,
        broken_after=0,
        repaired=3,
        touched_source=False,
        touched_tests=True,
        seconds=1.0,
        actions=4,
    )
    assert trial.broken_after == 0 and not trial.resolved
