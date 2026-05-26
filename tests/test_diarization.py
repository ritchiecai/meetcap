"""tests for speaker diarization services"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from meetcap.services.diarization import (
    DiarizationSegment,
    DiarizationService,
    SherpaOnnxDiarizationService,
    _merge_small_clusters,
    assign_speakers,
)
from meetcap.services.transcription import TranscriptSegment


class TestDiarizationSegment:
    """test diarization segment dataclass"""

    def test_create_segment(self):
        """test creating a diarization segment"""
        seg = DiarizationSegment(start=1.0, end=5.0, speaker=0)

        assert seg.start == 1.0
        assert seg.end == 5.0
        assert seg.speaker == 0


class TestDiarizationService:
    """test base diarization service"""

    def test_diarize_not_implemented(self):
        """test that base class raises NotImplementedError"""
        service = DiarizationService()
        with pytest.raises(NotImplementedError):
            service.diarize(Path("test.wav"))

    def test_load_model_not_implemented(self):
        """test that base class raises NotImplementedError"""
        service = DiarizationService()
        with pytest.raises(NotImplementedError):
            service.load_model()

    def test_unload_model_not_implemented(self):
        """test that base class raises NotImplementedError"""
        service = DiarizationService()
        with pytest.raises(NotImplementedError):
            service.unload_model()


class TestSherpaOnnxDiarizationService:
    """test sherpa-onnx diarization service"""

    def test_init(self):
        """test initialization"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/path/to/seg.onnx",
            embedding_model="/path/to/emb.onnx",
            num_speakers=3,
            threshold=0.9,
        )

        assert service.segmentation_model == "/path/to/seg.onnx"
        assert service.embedding_model == "/path/to/emb.onnx"
        assert service.num_speakers == 3
        assert service.threshold == 0.9
        assert service.sd is None

    def test_init_defaults(self):
        """test default parameters"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )

        assert service.num_speakers == -1
        assert service.threshold == 0.90
        assert service.min_duration_on == 0.3
        assert service.min_duration_off == 0.5
        # apple silicon defaults: cpu provider, 4 intra-op threads.
        # see services/diarization.py for measurement-based rationale.
        assert service.provider == "cpu"
        assert service.num_threads == 4

    def test_init_custom_provider_and_threads(self):
        """provider/num_threads can be overridden (e.g. coreml on x86)"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
            provider="coreml",
            num_threads=8,
        )
        assert service.provider == "coreml"
        assert service.num_threads == 8

    def test_load_model_import_error(self):
        """test handling import error when sherpa-onnx not installed"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )

        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "sherpa_onnx":
                raise ImportError("No module named 'sherpa_onnx'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc:
                service.load_model()

            assert "sherpa-onnx not installed" in str(exc.value)

    def test_load_model_missing_segmentation(self, tmp_path):
        """test error when segmentation model file missing"""
        emb_path = tmp_path / "emb.onnx"
        emb_path.write_bytes(b"fake")

        service = SherpaOnnxDiarizationService(
            segmentation_model=str(tmp_path / "missing_seg.onnx"),
            embedding_model=str(emb_path),
        )

        mock_sherpa = Mock()
        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            with pytest.raises(FileNotFoundError) as exc:
                service.load_model()
            assert "segmentation model not found" in str(exc.value)

    def test_load_model_missing_embedding(self, tmp_path):
        """test error when embedding model file missing"""
        seg_path = tmp_path / "seg.onnx"
        seg_path.write_bytes(b"fake")

        service = SherpaOnnxDiarizationService(
            segmentation_model=str(seg_path),
            embedding_model=str(tmp_path / "missing_emb.onnx"),
        )

        mock_sherpa = Mock()
        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            with pytest.raises(FileNotFoundError) as exc:
                service.load_model()
            assert "embedding model not found" in str(exc.value)

    def test_load_model_success(self, tmp_path):
        """test successful model loading"""
        seg_path = tmp_path / "seg.onnx"
        seg_path.write_bytes(b"fake")
        emb_path = tmp_path / "emb.onnx"
        emb_path.write_bytes(b"fake")

        service = SherpaOnnxDiarizationService(
            segmentation_model=str(seg_path),
            embedding_model=str(emb_path),
        )

        mock_sherpa = Mock()
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_sherpa.OfflineSpeakerDiarizationConfig.return_value = mock_config
        mock_sd = Mock()
        mock_sherpa.OfflineSpeakerDiarization.return_value = mock_sd

        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            service.load_model()

        assert service.sd is mock_sd

    def test_load_model_passes_provider_and_threads(self, tmp_path):
        """provider/num_threads must be forwarded to both sherpa-onnx
        sub-config classes (segmentation + embedding). regression guard
        for the 2026-05-25 perf fix."""
        seg_path = tmp_path / "seg.onnx"
        seg_path.write_bytes(b"fake")
        emb_path = tmp_path / "emb.onnx"
        emb_path.write_bytes(b"fake")

        service = SherpaOnnxDiarizationService(
            segmentation_model=str(seg_path),
            embedding_model=str(emb_path),
            provider="coreml",
            num_threads=2,
        )

        mock_sherpa = Mock()
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_sherpa.OfflineSpeakerDiarizationConfig.return_value = mock_config
        mock_sherpa.OfflineSpeakerDiarization.return_value = Mock()

        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            service.load_model()

        seg_kwargs = mock_sherpa.OfflineSpeakerSegmentationModelConfig.call_args.kwargs
        emb_kwargs = mock_sherpa.SpeakerEmbeddingExtractorConfig.call_args.kwargs
        assert seg_kwargs.get("provider") == "coreml"
        assert seg_kwargs.get("num_threads") == 2
        assert emb_kwargs.get("provider") == "coreml"
        assert emb_kwargs.get("num_threads") == 2

    def test_load_model_idempotent(self, tmp_path):
        """test that loading twice doesn't reload"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
        service.sd = Mock()
        service.load_model()  # should not attempt to reload

    def test_load_model_config_validation_failure(self, tmp_path):
        """test error when sherpa-onnx config validation fails"""
        seg_path = tmp_path / "seg.onnx"
        seg_path.write_bytes(b"fake")
        emb_path = tmp_path / "emb.onnx"
        emb_path.write_bytes(b"fake")

        service = SherpaOnnxDiarizationService(
            segmentation_model=str(seg_path),
            embedding_model=str(emb_path),
        )

        mock_sherpa = Mock()
        mock_config = Mock()
        mock_config.validate.return_value = False
        mock_sherpa.OfflineSpeakerDiarizationConfig.return_value = mock_config

        with patch.dict("sys.modules", {"sherpa_onnx": mock_sherpa}):
            with pytest.raises(RuntimeError) as exc:
                service.load_model()
            assert "config validation failed" in str(exc.value)

    def test_diarize_file_not_found(self, tmp_path):
        """test diarization with missing file"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
        with pytest.raises(FileNotFoundError):
            service.diarize(tmp_path / "nonexistent.wav")

    def test_diarize_success_no_resample(self, tmp_path):
        """test successful diarization without resampling"""
        import numpy as np

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake audio")

        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )

        # set up mock sd engine
        mock_sd = Mock()
        mock_sd.sample_rate = 16000

        # mock result segments
        mock_segment_0 = Mock()
        mock_segment_0.start = 0.0
        mock_segment_0.end = 3.0
        mock_segment_0.speaker = 0

        mock_segment_1 = Mock()
        mock_segment_1.start = 3.0
        mock_segment_1.end = 6.0
        mock_segment_1.speaker = 1

        mock_result = Mock()
        mock_result.sort_by_start_time.return_value = [mock_segment_0, mock_segment_1]
        mock_sd.process.return_value = mock_result

        service.sd = mock_sd

        # mock soundfile to return audio at the same sample rate as sd
        audio_data = np.zeros((16000 * 6, 1), dtype=np.float32)
        mock_sf = Mock()
        mock_sf.read.return_value = (audio_data, 16000)

        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            segments = service.diarize(audio_path)

        assert len(segments) == 2
        assert segments[0].start == 0.0
        assert segments[0].end == 3.0
        assert segments[0].speaker == 0
        assert segments[1].speaker == 1

        # verify soundfile was called correctly
        mock_sf.read.assert_called_once_with(str(audio_path), dtype="float32", always_2d=True)

    def test_diarize_with_resampling(self, tmp_path):
        """test diarization when audio needs resampling"""
        import numpy as np

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake audio")

        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )

        mock_sd = Mock()
        mock_sd.sample_rate = 16000

        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.speaker = 0

        mock_result = Mock()
        mock_result.sort_by_start_time.return_value = [mock_segment]
        mock_sd.process.return_value = mock_result

        service.sd = mock_sd

        # audio at 48kHz (different from sd.sample_rate=16kHz), so resampling is needed
        audio_data = np.zeros((48000 * 2, 1), dtype=np.float32)
        mock_sf = Mock()
        mock_sf.read.return_value = (audio_data, 48000)

        resampled_audio = np.zeros(16000 * 2, dtype=np.float32)
        mock_librosa = Mock()
        mock_librosa.resample.return_value = resampled_audio

        with patch.dict("sys.modules", {"soundfile": mock_sf, "librosa": mock_librosa}):
            segments = service.diarize(audio_path)

        assert len(segments) == 1
        assert segments[0].start == 0.0
        assert segments[0].end == 2.0
        assert segments[0].speaker == 0

        # verify librosa was called for resampling
        mock_librosa.resample.assert_called_once()
        call_kwargs = mock_librosa.resample.call_args
        assert call_kwargs[1]["orig_sr"] == 48000
        assert call_kwargs[1]["target_sr"] == 16000

    def test_diarize_console_output(self, tmp_path):
        """test that diarize prints correct console messages"""
        import numpy as np

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake audio")

        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )

        mock_sd = Mock()
        mock_sd.sample_rate = 16000

        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.speaker = 0

        mock_result = Mock()
        mock_result.sort_by_start_time.return_value = [mock_segment]
        mock_sd.process.return_value = mock_result

        service.sd = mock_sd

        audio_data = np.zeros((16000 * 5, 1), dtype=np.float32)
        mock_sf = Mock()
        mock_sf.read.return_value = (audio_data, 16000)

        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            with patch("meetcap.services.diarization.console") as mock_console:
                service.diarize(audio_path)

        # verify console output includes the filename
        console_calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("test.wav" in call for call in console_calls)
        # verify completion message
        assert any("diarization complete" in call for call in console_calls)

    def test_diarize_calls_load_model(self, tmp_path):
        """test that diarize calls load_model if not already loaded"""
        import numpy as np

        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"fake audio")

        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )

        # sd is None initially, so diarize should call load_model
        mock_sd = Mock()
        mock_sd.sample_rate = 16000
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.speaker = 0
        mock_result = Mock()
        mock_result.sort_by_start_time.return_value = [mock_segment]
        mock_sd.process.return_value = mock_result

        audio_data = np.zeros((16000, 1), dtype=np.float32)
        mock_sf = Mock()
        mock_sf.read.return_value = (audio_data, 16000)

        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            with patch.object(service, "load_model") as mock_load:
                # after load_model is called, sd should be set
                def set_sd():
                    service.sd = mock_sd

                mock_load.side_effect = set_sd
                segments = service.diarize(audio_path)

        mock_load.assert_called_once()
        assert len(segments) == 1

    def test_unload_model(self):
        """test model unloading"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
        service.sd = Mock()
        service.unload_model()
        assert service.sd is None

    def test_unload_model_when_not_loaded(self):
        """test unloading when model not loaded"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
        service.unload_model()  # should not raise
        assert service.sd is None

    def test_unload_model_calls_gc_collect(self):
        """test that unload_model calls gc.collect for memory cleanup"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
        service.sd = Mock()

        with patch("gc.collect") as mock_gc:
            service.unload_model()

        mock_gc.assert_called_once()
        assert service.sd is None

    def test_unload_model_tries_mlx_clear_cache(self):
        """test that unload_model tries to clear mlx metal cache"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
        service.sd = Mock()

        mock_mx = Mock()
        with patch("gc.collect"):
            with patch.dict("sys.modules", {"mlx": Mock(), "mlx.core": mock_mx}):
                service.unload_model()

        assert service.sd is None

    def test_unload_model_ignores_mlx_import_error(self):
        """test that unload_model handles missing mlx gracefully"""
        service = SherpaOnnxDiarizationService(
            segmentation_model="/seg.onnx",
            embedding_model="/emb.onnx",
        )
        service.sd = Mock()

        with patch("gc.collect"):
            # mlx is not installed, should not raise
            service.unload_model()

        assert service.sd is None


class TestAssignSpeakers:
    """test speaker assignment algorithm"""

    def test_single_speaker(self):
        """test with single speaker covering all segments"""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=2.0, text="Hello"),
            TranscriptSegment(id=1, start=2.0, end=4.0, text="World"),
        ]
        diar = [DiarizationSegment(start=0.0, end=5.0, speaker=0)]

        result_segments, speakers = assign_speakers(segments, diar)

        assert result_segments[0].speaker_id == 0
        assert result_segments[1].speaker_id == 0
        assert len(speakers) == 1
        assert speakers[0]["label"] == "Speaker 1"

    def test_multiple_speakers(self):
        """test with clear speaker boundaries"""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=3.0, text="Hello from A"),
            TranscriptSegment(id=1, start=5.0, end=8.0, text="Hello from B"),
            TranscriptSegment(id=2, start=10.0, end=13.0, text="Back to A"),
        ]
        diar = [
            DiarizationSegment(start=0.0, end=4.0, speaker=0),
            DiarizationSegment(start=4.5, end=9.0, speaker=1),
            DiarizationSegment(start=9.5, end=14.0, speaker=0),
        ]

        result_segments, speakers = assign_speakers(segments, diar)

        assert result_segments[0].speaker_id == 0
        assert result_segments[1].speaker_id == 1
        assert result_segments[2].speaker_id == 0
        assert len(speakers) == 2

    def test_non_sequential_speaker_ids(self):
        """test remapping of non-sequential speaker IDs from sherpa-onnx"""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=3.0, text="First"),
            TranscriptSegment(id=1, start=5.0, end=8.0, text="Second"),
            TranscriptSegment(id=2, start=10.0, end=13.0, text="Third"),
        ]
        diar = [
            DiarizationSegment(start=0.0, end=4.0, speaker=0),
            DiarizationSegment(start=4.5, end=9.0, speaker=5),  # non-sequential
            DiarizationSegment(start=9.5, end=14.0, speaker=12),  # non-sequential
        ]

        result_segments, speakers = assign_speakers(segments, diar)

        # should be remapped to 0, 1, 2
        assert result_segments[0].speaker_id == 0
        assert result_segments[1].speaker_id == 1
        assert result_segments[2].speaker_id == 2
        assert len(speakers) == 3
        assert speakers[0]["id"] == 0
        assert speakers[1]["id"] == 1
        assert speakers[2]["id"] == 2

    def test_empty_diarization(self):
        """test with no diarization segments"""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=2.0, text="Hello"),
        ]
        diar = []

        result_segments, speakers = assign_speakers(segments, diar)

        assert result_segments[0].speaker_id is None
        assert speakers == []

    def test_overlapping_diarization(self):
        """test with overlapping diarization segments"""
        segments = [
            TranscriptSegment(id=0, start=1.0, end=3.0, text="Overlap test"),
        ]
        # two speakers overlap, but speaker 1 has more overlap
        diar = [
            DiarizationSegment(start=0.0, end=1.5, speaker=0),  # 0.5s overlap
            DiarizationSegment(start=1.0, end=4.0, speaker=1),  # 2.0s overlap
        ]

        result_segments, speakers = assign_speakers(segments, diar)

        assert result_segments[0].speaker_id == 0  # remapped from 1 to 0
        # speaker 1 has maximum overlap, but after remapping it becomes 0
        # (only one unique speaker in the result since speaker 0 has less overlap)

    def test_no_overlap_assigns_none(self):
        """test segment with no overlapping diarization"""
        segments = [
            TranscriptSegment(id=0, start=10.0, end=12.0, text="No overlap"),
        ]
        diar = [
            DiarizationSegment(start=0.0, end=5.0, speaker=0),
        ]

        result_segments, speakers = assign_speakers(segments, diar)

        assert result_segments[0].speaker_id is None
        assert speakers == []

    def test_speaker_labels(self):
        """test speaker label generation"""
        segments = [
            TranscriptSegment(id=0, start=0.0, end=2.0, text="A"),
            TranscriptSegment(id=1, start=3.0, end=5.0, text="B"),
        ]
        diar = [
            DiarizationSegment(start=0.0, end=2.5, speaker=0),
            DiarizationSegment(start=2.5, end=5.5, speaker=1),
        ]

        _, speakers = assign_speakers(segments, diar)

        assert speakers[0] == {"id": 0, "label": "Speaker 1"}
        assert speakers[1] == {"id": 1, "label": "Speaker 2"}


class TestMergeSmallClusters:
    """test post-clustering small-cluster merge"""

    def test_no_merge_when_all_large(self):
        """test no merge when all clusters are above threshold."""
        segments = [
            DiarizationSegment(start=0, end=5, speaker=0),
            DiarizationSegment(start=5, end=10, speaker=0),
            DiarizationSegment(start=10, end=15, speaker=0),
            DiarizationSegment(start=15, end=20, speaker=1),
            DiarizationSegment(start=20, end=25, speaker=1),
            DiarizationSegment(start=25, end=30, speaker=1),
        ]
        result = _merge_small_clusters(segments, min_segments=3)
        speakers = {s.speaker for s in result}
        assert len(speakers) == 2

    def test_merge_small_cluster_into_nearest(self):
        """test that small clusters merge into the temporally nearest large cluster."""
        segments = [
            # speaker 0: 5 segments (large)
            DiarizationSegment(start=0, end=5, speaker=0),
            DiarizationSegment(start=10, end=15, speaker=0),
            DiarizationSegment(start=20, end=25, speaker=0),
            DiarizationSegment(start=30, end=35, speaker=0),
            DiarizationSegment(start=40, end=45, speaker=0),
            # speaker 1: 5 segments (large)
            DiarizationSegment(start=5, end=10, speaker=1),
            DiarizationSegment(start=15, end=20, speaker=1),
            DiarizationSegment(start=25, end=30, speaker=1),
            DiarizationSegment(start=35, end=40, speaker=1),
            DiarizationSegment(start=45, end=50, speaker=1),
            # speaker 2: 1 segment (small, should be merged)
            DiarizationSegment(start=22, end=24, speaker=2),
        ]
        result = _merge_small_clusters(segments, min_segments=3)
        speakers = {s.speaker for s in result}
        assert len(speakers) == 2
        # the small segment near t=23 should have been assigned to speaker 0 or 1
        small_seg = [s for s in result if s.start == 22][0]
        assert small_seg.speaker in (0, 1)

    def test_no_merge_on_empty(self):
        """test empty segment list."""
        assert _merge_small_clusters([], min_segments=3) == []

    def test_no_merge_when_all_small(self):
        """test no merge when ALL clusters are small (don't destroy everything)."""
        segments = [
            DiarizationSegment(start=0, end=5, speaker=0),
            DiarizationSegment(start=5, end=10, speaker=1),
            DiarizationSegment(start=10, end=15, speaker=2),
        ]
        result = _merge_small_clusters(segments, min_segments=3)
        # should not merge — all clusters are small, merging would be destructive
        speakers = {s.speaker for s in result}
        assert len(speakers) == 3

    def test_simulated_long_interview_over_segmentation(self):
        """test fixing a 2-person interview that was split into 6 speakers."""
        segments = []
        # speaker 0: 100 segments (real)
        for i in range(100):
            segments.append(DiarizationSegment(start=i * 30.0, end=i * 30.0 + 10, speaker=0))
        # speaker 1: 90 segments (real)
        for i in range(90):
            segments.append(DiarizationSegment(start=i * 30.0 + 15, end=i * 30.0 + 25, speaker=1))
        # spurious speakers: 1-2 segments each
        segments.append(DiarizationSegment(start=300, end=305, speaker=2))
        segments.append(DiarizationSegment(start=600, end=608, speaker=3))
        segments.append(DiarizationSegment(start=1800, end=1804, speaker=4))
        segments.append(DiarizationSegment(start=2700, end=2706, speaker=5))

        segments.sort(key=lambda s: s.start)
        total = len(segments)
        min_segs = max(3, int(total * 0.05))

        result = _merge_small_clusters(segments, min_segs)
        speakers = {s.speaker for s in result}
        assert len(speakers) == 2
