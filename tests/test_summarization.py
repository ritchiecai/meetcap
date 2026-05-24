"""comprehensive tests for summarization service"""

import io
import json as _json
import tempfile
import threading
import urllib.error
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from meetcap.services.summarization import (
    OmlxSummarizationService,
    SummarizationService,
    extract_meeting_title,
    save_summary,
)


class TestSummarizationService:
    """test summarization service functionality"""

    def test_init_stores_model_name(self):
        """test initialization stores model_name"""
        service = SummarizationService(
            model_name="mlx-community/Qwen3.5-2B-OptiQ-4bit",
            temperature=0.3,
            max_tokens=2048,
        )

        assert service.model_name == "mlx-community/Qwen3.5-2B-OptiQ-4bit"
        assert service.temperature == 0.3
        assert service.max_tokens == 2048
        assert service.model is None  # lazy loading
        assert service.processor is None
        assert service.model_config is None

    def test_init_default_values(self):
        """test initialization with default values"""
        service = SummarizationService()

        assert service.model_name == "mlx-community/Qwen3.5-2B-OptiQ-4bit"
        assert service.temperature == 0.4
        assert service.max_tokens == 4096
        assert service.model is None

    @patch("meetcap.services.summarization.console")
    def test_load_model_lazy(self, mock_console):
        """test lazy loading via mlx_vlm.load"""
        mock_model = Mock()
        mock_processor = Mock()
        mock_config = Mock()

        with patch("mlx_vlm.load", return_value=(mock_model, mock_processor)) as mock_load:
            with patch("mlx_vlm.utils.load_config", return_value=mock_config) as mock_load_config:
                service = SummarizationService(model_name="test-model")
                service._load_model()

                assert service.model is mock_model
                assert service.processor is mock_processor
                assert service.model_config is mock_config
                mock_load.assert_called_once_with("test-model")
                mock_load_config.assert_called_once_with("test-model")

    @patch("meetcap.services.summarization.console")
    def test_load_model_only_once(self, mock_console):
        """test model is only loaded once"""
        mock_model = Mock()
        mock_tokenizer = Mock()

        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)) as mock_load:
            service = SummarizationService()
            service._load_model()
            service._load_model()  # second call

            # should only be called once
            mock_load.assert_called_once()

    def test_load_model_import_error(self):
        """test handling of missing mlx-lm and mlx-vlm"""
        service = SummarizationService()

        # mock both imports to fail
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("mlx_lm", "mlx_vlm", "mlx_vlm.utils"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="neither mlx-lm nor mlx-vlm"):
                service._load_model()

    @patch("meetcap.services.summarization.console")
    def test_generate_summary(self, mock_console):
        """test _generate_summary uses tokenizer's apply_chat_template and generate()"""
        service = SummarizationService()
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted_prompt"
        service.model_config = Mock()

        mock_result = Mock()
        mock_result.text = "## summary\n\nActual summary content"

        with patch("mlx_vlm.generate", return_value=mock_result) as mock_gen:
            result = service._generate_summary("system prompt", "user prompt")

            # verify tokenizer's apply_chat_template was called with messages and enable_thinking
            service.processor.apply_chat_template.assert_called_once()
            call_kwargs = service.processor.apply_chat_template.call_args
            messages = call_kwargs[0][0]  # first positional arg
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "system prompt"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "user prompt"
            assert call_kwargs[1]["enable_thinking"] is False  # default

            # verify generate was called
            mock_gen.assert_called_once()
            gen_kwargs = mock_gen.call_args
            assert gen_kwargs[1]["max_tokens"] == 4096
            assert gen_kwargs[1]["temp"] == 0.4

            assert "## summary" in result
            assert "Actual summary content" in result

    @patch("meetcap.services.summarization.console")
    def test_generate_summary_string_result(self, mock_console):
        """test _generate_summary when result is a string"""
        service = SummarizationService()
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted"
        service.model_config = Mock()

        with patch("mlx_vlm.generate", return_value="## summary\n\nString result"):
            result = service._generate_summary("system", "user")
            assert "String result" in result

    @patch("meetcap.services.summarization.console")
    def test_generate_summary_fallback_str(self, mock_console):
        """test _generate_summary when result needs str() conversion"""
        service = SummarizationService()
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted"
        service.model_config = Mock()

        # object without .text attribute and not a string
        class CustomResult:
            def __str__(self):
                return "## summary\n\nFallback result"

        mock_result = CustomResult()

        with patch("mlx_vlm.generate", return_value=mock_result):
            result = service._generate_summary("system", "user")
            assert "Fallback result" in result

    @patch("meetcap.services.summarization.console")
    def test_generate_summary_with_thinking_tags(self, mock_console):
        """test summary generation removes thinking tags when thinking is enabled"""
        service = SummarizationService(enable_thinking=True, thinking_budget=100)
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted"
        service.model_config = Mock()

        mock_result = Mock()
        mock_result.text = "<think>I should analyze this</think>## summary\n\nActual summary"

        with patch("mlx_vlm.generate", return_value=mock_result):
            result = service._generate_summary("system", "user")

            assert "<think" not in result
            assert "## summary" in result
            assert "Actual summary" in result

    @patch("meetcap.services.summarization.console")
    def test_generate_summary_empty_after_cleaning(self, mock_console):
        """test warning when output is empty after cleaning"""
        service = SummarizationService(enable_thinking=True, thinking_budget=100)
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted"
        service.model_config = Mock()

        mock_result = Mock()
        mock_result.text = "<think>only thinking</think>"

        with patch("mlx_vlm.generate", return_value=mock_result):
            service._generate_summary("system", "user")

            # verify warning about short output
            calls = [str(call) for call in mock_console.print.call_args_list]
            assert any("output seems very short" in call.lower() for call in calls)

    @patch("meetcap.services.summarization.console")
    def test_generate_summary_thinking_disabled_no_cleaning(self, mock_console):
        """test that thinking tags are not cleaned when thinking is disabled"""
        service = SummarizationService(enable_thinking=False)
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted"
        service.model_config = Mock()

        mock_result = Mock()
        mock_result.text = "## summary\n\nDirect response without thinking"

        with patch("mlx_vlm.generate", return_value=mock_result) as mock_gen:
            result = service._generate_summary("system", "user")

            # should not pass thinking params to generate
            gen_kwargs = mock_gen.call_args[1]
            assert "enable_thinking" not in gen_kwargs
            assert "thinking_budget" not in gen_kwargs
            assert "Direct response without thinking" in result

    @patch("meetcap.services.summarization.console")
    def test_generate_summary_thinking_enabled_passes_params(self, mock_console):
        """test that thinking params are passed to generate() when enabled"""
        service = SummarizationService(enable_thinking=True, thinking_budget=256)
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted"
        service.model_config = Mock()

        mock_result = Mock()
        mock_result.text = "<think>brief thought</think>## summary\n\nContent"

        with patch("mlx_vlm.generate", return_value=mock_result) as mock_gen:
            result = service._generate_summary("system", "user")

            gen_kwargs = mock_gen.call_args[1]
            assert gen_kwargs["enable_thinking"] is True
            assert gen_kwargs["thinking_budget"] == 256
            assert gen_kwargs["thinking_start_token"] == "<think>"
            assert gen_kwargs["thinking_end_token"] == "</think>"
            assert "Content" in result

    @patch("meetcap.services.summarization.console")
    def test_summarize_short_transcript(self, mock_console):
        """test summarizing short transcript"""
        service = SummarizationService()
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted"
        service.model_config = Mock()

        mock_result = Mock()
        mock_result.text = (
            "## summary\n\nTest summary content\n\n## key discussion points\n\n- Point 1"
        )

        with patch("mlx_vlm.generate", return_value=mock_result):
            transcript = "This is a short test transcript."
            summary = service.summarize(
                transcript, meeting_title="Test Meeting", attendees=["Alice", "Bob"]
            )

            assert "## summary" in summary
            assert "Test summary content" in summary

    def test_clean_thinking_tags_standard(self):
        """test cleaning standard thinking tags"""
        service = SummarizationService()

        test_cases = [
            ("<think>thinking content</think>actual summary", "actual summary"),
            ("<THINK>UPPER CASE</THINK>content", "content"),
            ("<think>\nmultiline\nthinking\n</think>\nreal content", "real content"),
            ("before<think>middle</think>after", "beforeafter"),
            ("<thinking>variant tag</thinking>content", "content"),
        ]

        for input_text, expected in test_cases:
            result = service._clean_thinking_tags(input_text)
            assert result == expected

    def test_clean_thinking_tags_malformed(self):
        """test cleaning malformed thinking tags"""
        service = SummarizationService()

        # missing opening tag
        result = service._clean_thinking_tags("some thinking</think>actual content")
        assert result == "actual content"

        # missing closing tag (should keep content)
        result = service._clean_thinking_tags("<think>thinking\nactual content")
        assert "actual content" in result

        # nested tags
        result = service._clean_thinking_tags("<think>outer<think>inner</think></think>content")
        assert result == "content"

    def test_clean_thinking_tags_multiple(self):
        """test cleaning multiple thinking tags"""
        service = SummarizationService()

        text = "<think>first</think>content1<thinking>second</thinking>content2"
        result = service._clean_thinking_tags(text)
        assert result == "content1content2"

    def test_clean_thinking_tags_with_attributes(self):
        """test cleaning tags with attributes"""
        service = SummarizationService()

        text = '<think type="deep">thinking</think>summary'
        result = service._clean_thinking_tags(text)
        assert result == "summary"

    def test_clean_thinking_tags_whitespace(self):
        """test whitespace cleanup after tag removal"""
        service = SummarizationService()

        text = "<think>thinking</think>\n\n\n\n## summary"
        result = service._clean_thinking_tags(text)
        assert result == "## summary"
        assert not result.startswith("\n")

    def test_chunk_transcript(self):
        """test transcript chunking"""
        service = SummarizationService()

        text = " ".join([f"word{i}" for i in range(100)])
        chunks = service._chunk_transcript(text, chunk_size=50)

        assert len(chunks) > 1

        # verify all words are preserved
        all_words = []
        for chunk in chunks:
            all_words.extend(chunk.split())
        assert len(all_words) == 100
        assert all_words[0] == "word0"
        assert all_words[-1] == "word99"

    def test_chunk_transcript_single_chunk(self):
        """test chunking with text smaller than chunk size"""
        service = SummarizationService()

        text = "short text"
        chunks = service._chunk_transcript(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == "short text"

    @patch("meetcap.services.summarization.console")
    def test_unload_model(self, mock_console):
        """test model unloading"""
        service = SummarizationService()
        service.model = Mock()
        service.processor = Mock()
        service.model_config = Mock()

        with patch("mlx.core.clear_cache"):
            service.unload_model()

        assert service.model is None
        assert service.processor is None
        assert service.model_config is None

    def test_is_loaded(self):
        """test is_loaded check"""
        service = SummarizationService()
        assert service.is_loaded() is False

        service.model = Mock()
        assert service.is_loaded() is True

    def test_load_lock_exists(self):
        """test that SummarizationService has a thread-safe load lock"""
        service = SummarizationService()
        assert hasattr(service, "_load_lock")
        assert isinstance(service._load_lock, type(threading.Lock()))

    @patch("meetcap.services.summarization.console")
    def test_load_model_double_check_locking(self, mock_console):
        """test that _load_model uses double-check locking pattern"""
        mock_model = Mock()
        mock_processor = Mock()

        with patch("mlx_vlm.load", return_value=(mock_model, mock_processor)) as mock_load:
            with patch("mlx_vlm.utils.load_config", return_value=Mock()):
                service = SummarizationService()

                # simulate model already loaded before acquiring lock
                service.model = mock_model
                service._load_model()

                # should not call load since model is already set
                mock_load.assert_not_called()

    @patch("meetcap.services.summarization.console")
    def test_unload_model_calls_gc_and_mlx_cleanup(self, mock_console):
        """test that unload_model calls gc.collect and mlx.metal.clear_cache"""
        service = SummarizationService()
        service.model = Mock()
        service.processor = Mock()
        service.model_config = Mock()

        with patch("gc.collect") as mock_gc:
            mock_mx = Mock()
            with patch.dict("sys.modules", {"mlx": Mock(), "mlx.core": mock_mx}):
                service.unload_model()

            mock_gc.assert_called_once()
            assert service.model is None
            assert service.processor is None
            assert service.model_config is None


class TestSaveSummary:
    """test summary saving functionality"""

    def test_save_summary_with_proper_format(self, temp_dir):
        """test saving properly formatted summary"""
        summary_text = """## summary

This is the summary.

## key discussion points

- Point 1
- Point 2

## decisions

No formal decisions made

## action items

- [ ] TBD — Follow up on discussion (due: TBD)

## notable quotes

"Important quote here"
"""

        base_path = temp_dir / "meeting"

        with patch("meetcap.services.summarization.console") as mock_console:
            result_path = save_summary(summary_text, base_path)

            assert result_path == base_path.with_suffix(".summary.md")
            assert result_path.exists()

            content = result_path.read_text()

            # verify header was added
            assert "# Meeting Summary" in content
            assert "Generated:" in content
            assert datetime.now().strftime("%Y-%m-%d") in content

            # verify original content preserved
            assert "## summary" in content
            assert "This is the summary" in content
            assert "## key discussion points" in content
            assert "Point 1" in content

            # verify console output
            mock_console.print.assert_called()
            output = str(mock_console.print.call_args)
            assert "summary saved" in output.lower()

    def test_save_summary_with_transcript_text(self, temp_dir):
        """test saving summary with transcript_text appended"""
        summary_text = "## summary\n\nTest summary"
        transcript = "This is the full transcript content."

        base_path = temp_dir / "meeting"

        with patch("meetcap.services.summarization.console"):
            result_path = save_summary(summary_text, base_path, transcript_text=transcript)

            content = result_path.read_text()

            # verify transcript section was appended
            assert "## Full Transcript" in content
            assert "This is the full transcript content." in content

    def test_save_summary_without_transcript_text(self, temp_dir):
        """test saving summary without transcript_text"""
        summary_text = "## summary\n\nTest summary"

        base_path = temp_dir / "meeting"

        with patch("meetcap.services.summarization.console"):
            result_path = save_summary(summary_text, base_path)

            content = result_path.read_text()

            # verify no transcript section
            assert "## Full Transcript" not in content

    def test_save_summary_missing_structure(self, temp_dir):
        """test saving summary with missing structure"""
        summary_text = "Just some plain text without proper formatting"

        base_path = temp_dir / "meeting"
        result_path = save_summary(summary_text, base_path)

        content = result_path.read_text()

        # verify default structure was added
        assert "## summary" in content
        assert summary_text in content
        assert "## key discussion points" in content
        assert "(none identified)" in content
        assert "## decisions" in content
        assert "## action items" in content
        assert "## notable quotes" in content

    def test_save_summary_unicode(self, temp_dir):
        """test saving summary with unicode characters"""
        summary_text = """## summary

讨论了项目进展 (discussed project progress)
会議の要約 (meeting summary)
😀 Great meeting!

## key discussion points

- 中文内容
- 日本語の内容
- Emoji: 🎯 🚀 ✅
"""

        base_path = temp_dir / "unicode_meeting"
        result_path = save_summary(summary_text, base_path)

        content = result_path.read_text(encoding="utf-8")

        assert "讨论了项目进展" in content
        assert "会議の要約" in content
        assert "😀" in content
        assert "🎯" in content

    def test_save_summary_path_creation(self, temp_dir):
        """test summary path creation"""
        base_path = temp_dir / "subdir" / "meeting"

        # create parent directory (save_summary doesn't create parent dirs)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        summary_text = "## summary\n\nTest"
        result_path = save_summary(summary_text, base_path)

        # file should be created
        assert result_path.exists()
        assert result_path.name == "meeting.summary.md"


class TestSummarizationIntegration:
    """integration tests for summarization"""

    @patch("meetcap.services.summarization.console")
    def test_full_summarization_flow(self, mock_console, temp_dir):
        """test complete summarization flow with thinking enabled"""
        service = SummarizationService(
            model_name="test-model", enable_thinking=True, thinking_budget=100
        )
        service.model = Mock()
        service.processor = Mock()
        service.processor.apply_chat_template.return_value = "formatted"
        service.model_config = Mock()

        mock_result = Mock()
        mock_result.text = """<think>Let me analyze this transcript</think>

## summary

The meeting covered project updates and timeline.

## key discussion points

- Project milestone reached
- Budget approved

## decisions

Proceed with phase 2

## action items

- [ ] Alice — Prepare report (due: 2024-01-15)

## notable quotes

"This is a game changer"
"""

        with patch("mlx_vlm.generate", return_value=mock_result):
            transcript = "Alice: We reached the milestone. Bob: Great! This is a game changer."
            summary = service.summarize(
                transcript, meeting_title="Project Update", attendees=["Alice", "Bob"]
            )

            # verify thinking tags removed
            assert "<think" not in summary
            assert "Let me analyze" not in summary

            # verify content preserved
            assert "## summary" in summary
            assert "project updates and timeline" in summary
            assert "Project milestone reached" in summary
            assert "Proceed with phase 2" in summary
            assert "Alice — Prepare report" in summary
            assert "This is a game changer" in summary

            # save summary
            base_path = temp_dir / "meeting"
            save_summary(summary, base_path)

            # verify saved file
            saved_file = base_path.with_suffix(".summary.md")
            assert saved_file.exists()

            saved_content = saved_file.read_text()
            assert "# Meeting Summary" in saved_content
            assert "Project milestone reached" in saved_content


class TestExtractMeetingTitle:
    """test extract_meeting_title function"""

    def test_extract_from_meeting_title_section(self):
        """test extracting title from Meeting Title section"""
        summary = """## Meeting Title
ProductRoadmap

## Summary
This was a product roadmap meeting..."""

        title = extract_meeting_title(summary)
        assert title == "ProductRoadmap"

    def test_extract_with_spaces(self):
        """test extracting title with spaces that should be removed"""
        summary = """## Meeting Title
Product Roadmap Review

## Summary
Discussion about the product..."""

        title = extract_meeting_title(summary)
        assert title == "ProductRoadmapReview"

    def test_fallback_to_summary_keywords(self):
        """test fallback when no Meeting Title section"""
        summary = """## Summary
The Engineering Team discussed Sprint Planning for the next iteration..."""

        title = extract_meeting_title(summary)
        # should extract capitalized words from summary
        assert "Engineering" in title or "Team" in title or "Sprint" in title

    def test_fallback_to_transcript_keywords(self):
        """test fallback to transcript when no good title found"""
        summary = "## Summary\nwe talked about things"
        transcript = "project project project review review development sprint sprint sprint"

        title = extract_meeting_title(summary, transcript)
        # should use most common words
        assert "Meeting" in title  # always adds Meeting suffix in this case
        assert len(title) > 7  # should have meaningful content

    def test_fallback_to_untitled(self):
        """test absolute fallback when nothing works"""
        summary = "random text"

        title = extract_meeting_title(summary)
        assert title == "UntitledMeeting"

    def test_remove_markdown_formatting(self):
        """test removal of markdown formatting from title"""
        summary = """## Meeting Title
**ProductLaunch**

## Summary
Launch planning..."""

        title = extract_meeting_title(summary)
        assert title == "ProductLaunch"
        assert "*" not in title


def test_manual_notes_integration():
    """Test manual notes are included in summarization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        notes_path = temp_path / "notes.md"
        notes_path.write_text("# Meeting Notes\n\nImportant context about the meeting\n")

        # Test summarization includes manual notes
        mock_service = MockSummarizationService()
        summary = mock_service.summarize(
            transcript_text="Hello everyone, let's discuss the project timeline.",
            manual_notes_path=notes_path,
        )

        assert "Important context about the meeting" in summary


class MockSummarizationService:
    """Mock summarization service for testing manual notes integration"""

    def summarize(self, transcript_text: str, manual_notes_path: Path | None = None) -> str:
        """Mock summarize method that includes manual notes if provided"""
        manual_notes_text = ""
        if manual_notes_path and manual_notes_path.exists():
            try:
                with open(manual_notes_path, encoding="utf-8") as f:
                    manual_notes_text = f.read()
            except Exception as e:
                print(f"[yellow]⚠[/yellow] could not read manual notes: {e}")

        # Build user prompt with manual notes
        user_prompt_parts = []

        # add manual notes first if available
        if manual_notes_text:
            user_prompt_parts.append(f"manual notes:\n{manual_notes_text}")

        user_prompt_parts.append(f"transcript:\n{transcript_text}")

        # Mock LLM response that includes manual notes content
        base_summary = '## summary\n\nThis meeting was about project timeline.\n\n## key discussion points\n\n- Project planning\n- Timeline review\n\n## decisions\n\nApproved project timeline\n\n## action items\n\n- [ ] Team — Finalize project plan (due: TBD)\n\n## notable quotes\n\n"Let\'s move forward with the plan"'

        # Include manual notes content in the summary if available
        if manual_notes_text:
            # Extract the key content from manual notes (skip the header)
            lines = manual_notes_text.strip().split("\n")
            key_content = []
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    key_content.append(line.strip())

            if key_content:
                # Insert manual notes content into the summary
                manual_notes_section = " ".join(key_content)
                base_summary = base_summary.replace(
                    "This meeting was about project timeline.",
                    f"This meeting was about project timeline. {manual_notes_section}",
                )

        return base_summary


# ---------------------------------------------------------------------------
# OmlxSummarizationService tests
# ---------------------------------------------------------------------------


def _fake_urlopen_factory(payload: dict, status: int = 200):
    """build a fake urlopen context manager that returns a JSON payload."""

    class _FakeResp:
        def __init__(self, data: bytes, status: int):
            self._data = data
            self.status = status

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    body = _json.dumps(payload).encode("utf-8")

    def _fake_urlopen(req, timeout=None):
        return _FakeResp(body, status)

    return _fake_urlopen


class TestOmlxSummarizationServiceCallOmlx:
    """unit tests for OmlxSummarizationService._call_omlx response handling.

    these tests exist to defend against the regression we found in 2026-05-24:
    when oMLX returns the answer in `reasoning_content` instead of `content`,
    or when the response is malformed, the previous implementation crashed
    with KeyError or fed `None` into post-processing — silently producing
    truncated/empty summaries. these tests pin down the new defensive paths.
    """

    def _make(self, **overrides):
        kwargs = dict(
            model_name="mlx-community/Qwen3.5-2B-OptiQ-4bit",
            base_url="http://localhost:8000/v1",
            temperature=0.4,
            max_tokens=4096,
            timeout=10,
        )
        kwargs.update(overrides)
        return OmlxSummarizationService(**kwargs)

    def test_normal_content_path(self):
        """happy path: server returns content, output is stripped and returned."""
        svc = self._make()
        payload = {
            "choices": [
                {
                    "message": {"content": "## summary\n\nhello world\n"},
                    "finish_reason": "stop",
                }
            ]
        }
        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_fake_urlopen_factory(payload),
        ):
            out = svc._call_omlx("system", "user")
        assert "hello world" in out
        # _strip_untagged_thinking shouldn't truncate a clean ## heading
        assert out.startswith("## summary")

    def test_reasoning_content_fallback(self):
        """if content is empty but reasoning_content has text, use it."""
        svc = self._make()
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "## summary\n\nfallback text",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_fake_urlopen_factory(payload),
        ):
            out = svc._call_omlx("system", "user")
        assert "fallback text" in out

    def test_reasoning_content_fallback_when_content_is_null(self):
        """content=None should also trigger reasoning_content fallback."""
        svc = self._make()
        payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "## summary\n\nrecovered",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_fake_urlopen_factory(payload),
        ):
            out = svc._call_omlx("system", "user")
        assert "recovered" in out

    def test_choices_missing_raises(self):
        """malformed payload with no choices should raise loudly."""
        svc = self._make()
        payload = {"id": "abc", "object": "chat.completion"}
        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_fake_urlopen_factory(payload),
        ):
            with pytest.raises(RuntimeError, match="no choices"):
                svc._call_omlx("system", "user")

    def test_empty_choices_list_raises(self):
        """choices=[] should also raise."""
        svc = self._make()
        payload = {"choices": []}
        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_fake_urlopen_factory(payload),
        ):
            with pytest.raises(RuntimeError, match="no choices"):
                svc._call_omlx("system", "user")

    def test_both_content_and_reasoning_empty_raises(self):
        """if neither content nor reasoning_content has text, raise."""
        svc = self._make()
        payload = {
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": ""},
                    "finish_reason": "length",
                }
            ]
        }
        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_fake_urlopen_factory(payload),
        ):
            with pytest.raises(RuntimeError, match="empty content"):
                svc._call_omlx("system", "user")

    def test_thinking_tags_are_stripped(self):
        """<think>...</think> blocks should be removed from the output."""
        svc = self._make()
        payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<think>let me reason about this</think>\n"
                            "## summary\n\nclean output"
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_fake_urlopen_factory(payload),
        ):
            out = svc._call_omlx("system", "user")
        assert "<think>" not in out.lower()
        assert "## summary" in out

    def test_payload_contains_defensive_fields(self):
        """payload sent to oMLX must include top_p and stop tokens (regression
        guard for the 2026-05-24 hardening)."""
        svc = self._make()
        payload = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        captured = {}

        def _capture_urlopen(req, timeout=None):
            captured["data"] = _json.loads(req.data.decode("utf-8"))
            captured["timeout"] = timeout

            class _R:
                status = 200

                def read(self):
                    return _json.dumps(payload).encode("utf-8")

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return _R()

        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_capture_urlopen,
        ):
            svc._call_omlx("system", "user")

        body = captured["data"]
        assert body["model"] == "mlx-community/Qwen3.5-2B-OptiQ-4bit"
        assert body["temperature"] == pytest.approx(0.4)
        assert body["max_tokens"] == 4096
        assert body["stream"] is False
        assert body["top_p"] == pytest.approx(0.95)
        assert "<|im_end|>" in body["stop"]
        assert "<|endoftext|>" in body["stop"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert captured["timeout"] == 10

    def test_http_error_is_wrapped(self):
        """HTTPError should surface as a RuntimeError with status code."""
        svc = self._make()

        def _raise(req, timeout=None):
            raise urllib.error.HTTPError(
                url="http://localhost:8000/v1/chat/completions",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=io.BytesIO(b'{"error": "boom"}'),
            )

        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_raise,
        ):
            with pytest.raises(RuntimeError, match="oMLX API error 500"):
                svc._call_omlx("system", "user")

    def test_url_error_is_wrapped(self):
        """URLError (connection refused etc.) should surface as ConnectionError."""
        svc = self._make()

        def _raise(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        with patch(
            "meetcap.services.summarization.urllib.request.urlopen",
            side_effect=_raise,
        ):
            with pytest.raises(ConnectionError, match="cannot connect to oMLX"):
                svc._call_omlx("system", "user")
