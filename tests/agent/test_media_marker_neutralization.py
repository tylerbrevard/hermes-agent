"""llama.cpp media-marker neutralization chokepoint regression tests (#108760).

When a llama-server endpoint's own randomized ``media_marker`` (exposed by
``/props``) is quoted into the transcript as plain text — e.g. a tool result
dumping the server config — every later turn 400s with
``{"message": "Failed to tokenize prompt"}``: the server splits the outgoing
prompt on that exact marker and expects one attached bitmap per piece, so a
marker with no image behind it fails in mtmd. All endpoints on the same server
share the marker, so failover cannot route around it and the session wedges.

The fix is owned at the send-time sanitization chokepoint, on the outgoing API
copy only: markers are neutralized, never escaped — genuine images travel as
structured ``image_url`` parts and the server inserts its own marker, so the
only text this matches is a marker quoted as plain text. The stored transcript
keeps the real bytes the tool returned.
"""

import copy
import json

from agent.message_sanitization import (
    _neutralize_media_markers,
    _sanitize_messages_media_markers,
)

# The exact marker shape from the issue's /props dump (randomized per process
# start on real servers; the random segment is [A-Za-z0-9_-]{24} there but the
# pattern tolerates any length ≥ 1 so a shorter server build also matches).
LIVE_MARKER = "<__media_CROZLhoWqzum1cQTzZQKMYA9gqWTHJ3E__>"
# Older builds spell the placeholder without a random segment.
LEGACY_MEDIA = "<__media__>"
LEGACY_IMAGE = "<__image__>"

# --------------------------------------------------------------------------- #
# _neutralize_media_markers: the pure text transform                            #
# --------------------------------------------------------------------------- #


def test_randomized_marker_in_tool_result_text_is_neutralized():
    """The exact /props shape from #108760 is replaced, not passed through."""
    text = f'{{"media_marker": "{LIVE_MARKER}"}}'
    out = _neutralize_media_markers(text)
    assert LIVE_MARKER not in out
    assert "[media-marker removed]" in out
    # The surrounding JSON text the model actually needs survives.
    assert '"media_marker"' in out


def test_legacy_fixed_marker_spellings_are_neutralized():
    """Older server builds use the fixed ``<__media__>`` / ``<__image__>`` spellings."""
    for marker in (LEGACY_MEDIA, LEGACY_IMAGE):
        out = _neutralize_media_markers(f"before {marker} after")
        assert marker not in out
        assert out == f"before [media-marker removed] after"


def test_multiple_markers_and_clean_text():
    """Every occurrence is replaced; clean text is a byte-identical no-op (cache-safe)."""
    assert _neutralize_media_markers("no markers here") == "no markers here"
    out = _neutralize_media_markers(f"a {LIVE_MARKER} b {LEGACY_MEDIA} c")
    assert out == "a [media-marker removed] b [media-marker removed] c"


def test_lookalike_but_shorter_suffix_is_left_alone():
    """The pattern needs the closing ``__>``; a truncated quote stays intact.

    A marker cut mid-quote (log truncation) no longer matches the server's
    split token, so leaving it is correct — it cannot wedge the session.
    """
    truncated = "<__media_CROZLhoWqzum1cQTz"
    assert _neutralize_media_markers(f"x {truncated} y") == f"x {truncated} y"


# --------------------------------------------------------------------------- #
# _sanitize_messages_media_markers: the message-list chokepoint                 #
# --------------------------------------------------------------------------- #


def _wedged_transcript() -> list:
    """Minimal transcript shape from #108760: a /props dump in a tool result."""
    return [
        {"role": "user", "content": "check the server config"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "curl http://host:8080/v1/props"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": f'{{"default_settings": {{}}, "media_marker": "{LIVE_MARKER}"}}',
        },
        {"role": "user", "content": "ok now summarize"},
    ]


def test_chokepoint_neutralizes_marker_in_tool_result_on_api_copy():
    """The wedge scenario: marker quoted in a tool result leaves the outgoing copy."""
    api_messages = _wedged_transcript()
    changed = _sanitize_messages_media_markers(api_messages)
    assert changed is True
    wire_text = api_messages[2]["content"]
    assert LIVE_MARKER not in wire_text
    assert "[media-marker removed]" in wire_text


def test_chokepoint_catches_marker_inside_tool_call_arguments():
    """deep=True: a marker quoted inside a tool_call argument JSON is caught too."""
    api_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": f'{{"command": "grep {LIVE_MARKER} /tmp/out"}}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    changed = _sanitize_messages_media_markers(api_messages)
    assert changed is True
    assert LIVE_MARKER not in api_messages[0]["tool_calls"][0]["function"]["arguments"]


def test_chokepoint_catches_marker_in_multipart_content_and_nested_fields():
    """Content-part text and nested non-core fields (reasoning_details) are swept."""
    api_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"the server said {LIVE_MARKER} in props"},
            ],
        },
        {
            "role": "assistant",
            "content": "seen",
            "reasoning_details": [{"summary": f"marker was {LEGACY_IMAGE} here"}],
        },
    ]
    changed = _sanitize_messages_media_markers(api_messages)
    assert changed is True
    assert LIVE_MARKER not in api_messages[0]["content"][0]["text"]
    assert LEGACY_IMAGE not in api_messages[1]["reasoning_details"][0]["summary"]


def test_chokepoint_is_a_noop_on_clean_messages():
    """Byte-identical no-op when no marker is present (prompt-cache safe)."""
    api_messages = _wedged_transcript()
    clean = copy.deepcopy(api_messages)
    clean[2]["content"] = '{"default_settings": {}, "media_marker": "<not-a-marker>"}'
    before = copy.deepcopy(clean)
    changed = _sanitize_messages_media_markers(clean)
    assert changed is False
    assert clean == before


def test_stored_transcript_is_not_the_api_copy():
    """Contract: the sanitizer mutates the per-call API copy in place; the caller
    (assemble_api_request) passes a copy, so persisted history keeps the real text.
    This documents the split — the fix must run on the wire copy only."""
    stored = _wedged_transcript()
    api_copy = copy.deepcopy(stored)
    _sanitize_messages_media_markers(api_copy)
    assert LIVE_MARKER in stored[2]["content"]
    assert LIVE_MARKER not in api_copy[2]["content"]


# --------------------------------------------------------------------------- #
# E2E wiring: the real assemble_api_request phase must neutralize the marker    #
# on the outgoing request copy (real AIAgent, temp HERMES_HOME — root rubric).  #
# --------------------------------------------------------------------------- #


def test_assemble_api_request_neutralizes_marker_end_to_end(tmp_path, monkeypatch):
    """The #108760 wedge, replayed through the real request-assembly phase: the
    /props dump sits in a persisted tool result; the outgoing request copy must
    not carry the bare marker, while the stored transcript keeps it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_request_assembly import assemble_api_request

    agent = AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                    model="test-model", provider="openai-compat", api_key="test",
                    base_url="http://127.0.0.1:1/v1", max_iterations=4,
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    try:
        _reset_per_turn_agent_state(agent)
        messages = _wedged_transcript()
        stored_snapshot = copy.deepcopy(messages)
        assembled = assemble_api_request(
            agent,
            messages=messages,
            current_turn_user_idx=len(messages) - 1,
            _ext_prefetch_cache=None,
            _plugin_user_context=None,
            moa_config=None,
            active_system_prompt=None,
            original_user_message="ok now summarize",
            pending_moa_prepared_request=None,
            request_logger=None,
        )
        request_messages = assembled.api_messages
        wire = json.dumps(request_messages, ensure_ascii=False)
        assert LIVE_MARKER not in wire, (
            "assemble_api_request sent the bare llama.cpp media marker — the server "
            "splits the prompt on it and 400s (Failed to tokenize prompt)"
        )
        assert "[media-marker removed]" in wire
        # The persisted transcript keeps the real bytes the tool returned.
        assert messages == stored_snapshot
    finally:
        agent._session_db.close()
