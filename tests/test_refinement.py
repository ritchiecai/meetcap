"""tests for the LLM-based transcript refinement service"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from meetcap.services.refinement import (
    Correction,
    RefinementService,
    _apply_corrections,
    _chunk_segments,
    _extract_json_array,
    backup_original_transcript,
    load_hotwords,
    save_corrections_log,
)
from meetcap.services.transcription import (
    TranscriptResult,
    TranscriptSegment,
    save_transcript,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seg(i: int, text: str, start: float = 0.0, end: float = 1.0) -> TranscriptSegment:
    return TranscriptSegment(id=i, start=start, end=end, text=text)


def _result(segments: list[TranscriptSegment]) -> TranscriptResult:
    return TranscriptResult(
        audio_path="/tmp/fake.wav",
        sample_rate=16000,
        language="en",
        segments=segments,
        duration=10.0,
        stt={"engine": "test"},
    )


# ---------------------------------------------------------------------------
# load_hotwords
# ---------------------------------------------------------------------------


class TestLoadHotwords:
    def test_inline_only(self):
        words = load_hotwords(hotwords_inline=["TCADP", "Cognito"])
        assert words == ["TCADP", "Cognito"]

    def test_dedupes_and_preserves_order(self):
        words = load_hotwords(
            hotwords_inline=["TCADP", "Cognito", "TCADP", "BlackHole"]
        )
        assert words == ["TCADP", "Cognito", "BlackHole"]

    def test_strips_whitespace_and_blanks(self):
        words = load_hotwords(hotwords_inline=["  TCADP  ", "", "Cognito"])
        assert words == ["TCADP", "Cognito"]

    def test_file_loading(self, tmp_path: Path):
        f = tmp_path / "hotwords.txt"
        f.write_text(
            "# this is a comment\n"
            "TCADP\n"
            "\n"
            "Cognito\n"
            "  BlackHole  \n"
            "# another comment\n",
            encoding="utf-8",
        )
        words = load_hotwords(hotwords_file=f)
        assert words == ["TCADP", "Cognito", "BlackHole"]

    def test_inline_and_file_merged(self, tmp_path: Path):
        f = tmp_path / "hotwords.txt"
        f.write_text("Cognito\nJucoin\n", encoding="utf-8")
        words = load_hotwords(
            hotwords_inline=["TCADP", "Cognito"],
            hotwords_file=f,
        )
        assert words == ["TCADP", "Cognito", "Jucoin"]

    def test_missing_file_returns_inline(self, tmp_path: Path):
        f = tmp_path / "does-not-exist.txt"
        words = load_hotwords(hotwords_inline=["TCADP"], hotwords_file=f)
        assert words == ["TCADP"]

    def test_empty(self):
        assert load_hotwords() == []


# ---------------------------------------------------------------------------
# _extract_json_array
# ---------------------------------------------------------------------------


class TestExtractJsonArray:
    def test_clean_array(self):
        assert _extract_json_array('[{"a": 1}]') == [{"a": 1}]

    def test_empty_array(self):
        assert _extract_json_array("[]") == []

    def test_empty_string(self):
        assert _extract_json_array("") == []

    def test_whitespace_only(self):
        assert _extract_json_array("   \n  ") == []

    def test_with_code_fence(self):
        raw = '```json\n[{"x": "y"}]\n```'
        assert _extract_json_array(raw) == [{"x": "y"}]

    def test_with_plain_fence(self):
        raw = '```\n[{"x": "y"}]\n```'
        assert _extract_json_array(raw) == [{"x": "y"}]

    def test_with_prefix_prose(self):
        raw = 'Here is the JSON output:\n[{"k": 1}, {"k": 2}]'
        assert _extract_json_array(raw) == [{"k": 1}, {"k": 2}]

    def test_with_suffix_prose(self):
        raw = '[{"k": 1}]\nLet me know if you need anything else.'
        assert _extract_json_array(raw) == [{"k": 1}]

    def test_brackets_inside_strings_dont_break_parser(self):
        raw = '[{"text": "hello [world]"}]'
        assert _extract_json_array(raw) == [{"text": "hello [world]"}]

    def test_returns_empty_for_non_array(self):
        assert _extract_json_array('{"k": 1}') == []

    def test_returns_empty_for_garbage(self):
        assert _extract_json_array("totally not json") == []


# ---------------------------------------------------------------------------
# _apply_corrections
# ---------------------------------------------------------------------------


class TestApplyCorrections:
    def test_simple_replacement(self):
        segs = [_seg(0, "we use Cognitor for auth")]
        items = [
            {
                "segment_id": 0,
                "original": "Cognitor",
                "corrected": "Cognito",
                "reason": "product name",
            }
        ]
        new_segs, applied = _apply_corrections(segs, items)
        assert new_segs[0].text == "we use Cognito for auth"
        assert len(applied) == 1
        assert applied[0].original == "Cognitor"
        assert applied[0].corrected == "Cognito"
        assert applied[0].segment_id == 0

    def test_preserves_metadata(self):
        original = TranscriptSegment(
            id=0, start=1.5, end=3.0, text="hi", speaker_id=2, confidence=0.9
        )
        segs = [original]
        items = [{"segment_id": 0, "original": "hi", "corrected": "hello"}]
        new_segs, _ = _apply_corrections(segs, items)
        assert new_segs[0].start == 1.5
        assert new_segs[0].end == 3.0
        assert new_segs[0].speaker_id == 2
        assert new_segs[0].confidence == 0.9
        assert new_segs[0].text == "hello"

    def test_skips_when_substring_missing(self):
        segs = [_seg(0, "hello world")]
        items = [{"segment_id": 0, "original": "xyz", "corrected": "abc"}]
        new_segs, applied = _apply_corrections(segs, items)
        assert new_segs[0].text == "hello world"
        assert applied == []

    def test_case_insensitive_fallback(self):
        segs = [_seg(0, "we use cognito for auth")]
        items = [{"segment_id": 0, "original": "Cognito", "corrected": "Cognito"}]
        # original==corrected after case-correction; should be skipped because
        # noop, but if corrected differs we keep original casing's location
        items = [{"segment_id": 0, "original": "Cognito", "corrected": "AWS Cognito"}]
        new_segs, applied = _apply_corrections(segs, items)
        assert "AWS Cognito" in new_segs[0].text
        assert len(applied) == 1

    def test_unknown_segment_id_skipped(self):
        segs = [_seg(0, "foo")]
        items = [{"segment_id": 99, "original": "foo", "corrected": "bar"}]
        new_segs, applied = _apply_corrections(segs, items)
        assert new_segs[0].text == "foo"
        assert applied == []

    def test_noop_correction_skipped(self):
        segs = [_seg(0, "foo bar")]
        items = [{"segment_id": 0, "original": "foo", "corrected": "foo"}]
        _, applied = _apply_corrections(segs, items)
        assert applied == []

    def test_multiple_corrections_same_segment(self):
        segs = [_seg(0, "we use Cognitor and TeaCadPee daily")]
        items = [
            {"segment_id": 0, "original": "Cognitor", "corrected": "Cognito"},
            {"segment_id": 0, "original": "TeaCadPee", "corrected": "TCADP"},
        ]
        new_segs, applied = _apply_corrections(segs, items)
        assert new_segs[0].text == "we use Cognito and TCADP daily"
        assert len(applied) == 2

    def test_malformed_item_skipped(self):
        segs = [_seg(0, "foo")]
        items = [
            {"segment_id": "not-an-int", "original": "foo", "corrected": "bar"},
            {"original": "foo", "corrected": "bar"},  # missing segment_id
            {"segment_id": 0, "corrected": "bar"},  # missing original
        ]
        new_segs, applied = _apply_corrections(segs, items)
        assert new_segs[0].text == "foo"
        assert applied == []


# ---------------------------------------------------------------------------
# _chunk_segments
# ---------------------------------------------------------------------------


class TestChunkSegments:
    def test_empty(self):
        assert _chunk_segments([]) == []

    def test_single_chunk_under_limits(self):
        segs = [_seg(i, "x" * 50) for i in range(10)]
        chunks = _chunk_segments(segs, segs_per_chunk=80, max_chars=10_000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_chunked_by_segment_count(self):
        segs = [_seg(i, "x") for i in range(25)]
        chunks = _chunk_segments(segs, segs_per_chunk=10, max_chars=100_000)
        assert [len(c) for c in chunks] == [10, 10, 5]

    def test_chunked_by_char_budget(self):
        # each segment is ~210 chars (200 text + 8 overhead). Budget=500 means
        # ~2 segments per chunk.
        segs = [_seg(i, "y" * 200) for i in range(7)]
        chunks = _chunk_segments(segs, segs_per_chunk=80, max_chars=500)
        # don't pin exact distribution, but must respect budget on every chunk
        for c in chunks:
            chars = sum(len(s.text) + 8 for s in c)
            assert chars <= 500 or len(c) == 1
        assert sum(len(c) for c in chunks) == 7


# ---------------------------------------------------------------------------
# RefinementService — integration via mocked LLM call
# ---------------------------------------------------------------------------


class TestRefinementServiceMlxLm:
    def _make_service(self, **overrides):
        kwargs = dict(
            model_name="fake/model",
            backend="mlx-lm",
            mode="diff",
            temperature=0.1,
            max_tokens=256,
            hotwords=["Cognito", "TCADP"],
        )
        kwargs.update(overrides)
        return RefinementService(**kwargs)

    def test_no_segments_returns_empty(self):
        svc = self._make_service()
        result = svc.refine(_result([]))
        assert result.refined_segments == []
        assert result.corrections == []

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            RefinementService(backend="mlx-lm", mode="bogus")

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError):
            RefinementService(backend="grpc", mode="diff")

    def test_refine_applies_llm_corrections(self):
        svc = self._make_service()
        segs = [_seg(0, "we use Cognitor for auth")]
        transcript = _result(segs)

        fake_llm_output = json.dumps(
            [
                {
                    "segment_id": 0,
                    "original": "Cognitor",
                    "corrected": "Cognito",
                    "reason": "AWS product name",
                }
            ]
        )
        with patch.object(svc, "_call_in_process", return_value=fake_llm_output):
            result = svc.refine(transcript)

        assert len(result.refined_segments) == 1
        assert result.refined_segments[0].text == "we use Cognito for auth"
        assert len(result.corrections) == 1
        assert result.corrections[0].corrected == "Cognito"

    def test_refine_with_no_corrections(self):
        svc = self._make_service()
        transcript = _result([_seg(0, "everything is fine here")])
        with patch.object(svc, "_call_in_process", return_value="[]"):
            result = svc.refine(transcript)
        assert result.corrections == []
        # segments returned should reference originals (no rewrite needed)
        assert result.refined_segments[0].text == "everything is fine here"

    def test_refine_handles_chunk_failure_gracefully(self):
        svc = self._make_service(segs_per_chunk=2, max_chars_per_chunk=10_000)
        segs = [_seg(i, f"chunk {i}") for i in range(5)]
        transcript = _result(segs)

        call_count = {"n": 0}

        def fake(_sys, _user):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("boom")
            return "[]"

        with patch.object(svc, "_call_in_process", side_effect=fake):
            result = svc.refine(transcript)

        assert result.skipped_chunks == 1
        # transcript should still be intact
        assert [s.text for s in result.refined_segments] == [
            "chunk 0",
            "chunk 1",
            "chunk 2",
            "chunk 3",
            "chunk 4",
        ]


# ---------------------------------------------------------------------------
# save_corrections_log + backup_original_transcript
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_corrections_log(self, tmp_path: Path):
        base = tmp_path / "recording"
        corrections = [
            Correction(segment_id=0, original="Cognitor", corrected="Cognito", reason="r1"),
            Correction(segment_id=2, original="x", corrected="y"),
        ]
        out = save_corrections_log(corrections, base, metadata={"backend": "mlx-lm"})
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["count"] == 2
        assert data["metadata"]["backend"] == "mlx-lm"
        assert data["corrections"][0]["original"] == "Cognitor"
        assert data["corrections"][0]["corrected"] == "Cognito"

    def test_backup_creates_original_copies(self, tmp_path: Path):
        base = tmp_path / "recording"
        # produce real transcript files via save_transcript
        result = _result([_seg(0, "hello world")])
        save_transcript(result, base)

        txt_backup, json_backup = backup_original_transcript(base)
        assert txt_backup is not None
        assert json_backup is not None
        assert txt_backup.exists()
        assert json_backup.exists()
        assert "hello world" in txt_backup.read_text(encoding="utf-8")

    def test_backup_no_op_when_originals_missing(self, tmp_path: Path):
        base = tmp_path / "recording"
        txt_backup, json_backup = backup_original_transcript(base)
        assert txt_backup is None
        assert json_backup is None

    def test_backup_idempotent(self, tmp_path: Path):
        base = tmp_path / "recording"
        result = _result([_seg(0, "first version")])
        save_transcript(result, base)
        backup_original_transcript(base)

        # mutate transcript and try to back up again — should NOT overwrite
        result2 = _result([_seg(0, "refined version")])
        save_transcript(result2, base)
        txt_backup, json_backup = backup_original_transcript(base)
        # second call returns None because backup already exists
        assert txt_backup is None
        assert json_backup is None

        original_txt = base.with_suffix(".transcript.original.txt")
        assert "first version" in original_txt.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigRefinementSection:
    def test_default_disabled(self, tmp_path: Path):
        from meetcap.utils.config import Config

        cfg = Config(tmp_path / "config.toml")
        assert cfg.get("refinement", "enabled") is False
        assert cfg.get("refinement", "mode") == "diff"
        assert cfg.get("refinement", "keep_original") is True
        assert cfg.get("refinement", "preserve_filler_words") is True

    def test_env_override_enabled(self, tmp_path: Path, monkeypatch):
        from meetcap.utils.config import Config

        monkeypatch.setenv("MEETCAP_REFINEMENT_ENABLED", "true")
        monkeypatch.setenv("MEETCAP_REFINEMENT_MODE", "full")
        cfg = Config(tmp_path / "config.toml")
        assert cfg.get("refinement", "enabled") is True
        assert cfg.get("refinement", "mode") == "full"
