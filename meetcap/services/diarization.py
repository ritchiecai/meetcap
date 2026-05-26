"""speaker diarization service using sherpa-onnx"""

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# sherpa-onnx FastClustering uses cosine distance between speaker embeddings.
# higher threshold → fewer clusters (speakers); lower → more clusters.
# 0.85 (old default) over-segments on real audio (5 speakers instead of 4).
# 0.90 empirically gives the correct count on 4-speaker test data, and avoids
# the severe over-segmentation the user saw on 2-person interviews.
DEFAULT_CLUSTER_THRESHOLD = 0.90

# minimum fraction of total segments a speaker must have to survive
# post-clustering cleanup.  clusters below this fraction are merged into the
# temporally nearest larger speaker, mimicking pyannote 3.1's min_cluster_size.
# only activates on recordings with enough segments (≥30) to be meaningful.
MIN_CLUSTER_FRACTION = 0.05  # 5% of total segments
MIN_CLUSTER_FLOOR = 3  # absolute minimum (when fraction applies)
MIN_TOTAL_SEGMENTS_FOR_MERGE = 30  # don't merge on very short audio


@dataclass
class DiarizationSegment:
    """a time segment with speaker identity"""

    start: float
    end: float
    speaker: int  # 0-indexed speaker ID


class DiarizationService:
    """base class for speaker diarization services"""

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        """identify speakers in audio, return time-labeled segments."""
        raise NotImplementedError

    def load_model(self) -> None:
        """explicitly load model into memory."""
        raise NotImplementedError

    def unload_model(self) -> None:
        """explicitly unload model from memory."""
        raise NotImplementedError

    def is_loaded(self) -> bool:
        """check if model is currently loaded in memory."""
        return getattr(self, "sd", None) is not None


class SherpaOnnxDiarizationService(DiarizationService):
    """speaker diarization using sherpa-onnx with pyannote segmentation"""

    def __init__(
        self,
        segmentation_model: str,
        embedding_model: str,
        num_speakers: int = -1,
        threshold: float = DEFAULT_CLUSTER_THRESHOLD,
        min_duration_on: float = 0.3,
        min_duration_off: float = 0.5,
        provider: str = "cpu",
        num_threads: int = 4,
    ):
        """
        initialize sherpa-onnx diarization service.

        args:
            segmentation_model: path to pyannote ONNX segmentation model
            embedding_model: path to speaker embedding ONNX model
            num_speakers: expected number of speakers (-1 for auto-detect)
            threshold: clustering threshold (higher = fewer speakers)
            min_duration_on: minimum speech segment duration in seconds
            min_duration_off: minimum silence duration in seconds
            provider: onnxruntime execution provider for both segmentation
                and embedding ("cpu" or "coreml"). on apple silicon,
                empirically "cpu" + num_threads=4 outperforms coreml because
                embedding is invoked on many tiny (1~5s) chunks where ANE
                launch/copy overhead dominates. keep default "cpu" unless
                a specific deployment proves coreml is faster.
            num_threads: onnxruntime intra-op thread count for both
                segmentation and embedding. apple silicon m-series tops out
                at the number of performance cores (e.g., 4 on M4). values
                higher than P-core count usually hurt due to e-core
                contention.
        """
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.num_speakers = num_speakers
        self.threshold = threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.provider = provider
        self.num_threads = num_threads
        self.sd = None

    def load_model(self) -> None:
        """initialize the sherpa-onnx diarization engine."""
        if self.sd is not None:
            return

        try:
            import sherpa_onnx
        except ImportError as e:
            raise ImportError(
                "sherpa-onnx not installed. install with: pip install sherpa-onnx"
            ) from e

        # validate model paths exist
        if not Path(self.segmentation_model).exists():
            raise FileNotFoundError(
                f"segmentation model not found: {self.segmentation_model}. "
                f"run 'meetcap setup' to download diarization models."
            )
        if not Path(self.embedding_model).exists():
            raise FileNotFoundError(
                f"embedding model not found: {self.embedding_model}. "
                f"run 'meetcap setup' to download diarization models."
            )

        console.print("[cyan]loading diarization models...[/cyan]")

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=self.segmentation_model,
                ),
                provider=self.provider,
                num_threads=self.num_threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.embedding_model,
                provider=self.provider,
                num_threads=self.num_threads,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=self.num_speakers,
                threshold=self.threshold,
            ),
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

        if not config.validate():
            raise RuntimeError(
                "sherpa-onnx diarization config validation failed. "
                "check that model files are valid ONNX models."
            )

        self.sd = sherpa_onnx.OfflineSpeakerDiarization(config)
        console.print("[green]✓[/green] diarization models ready")

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        """
        run speaker diarization on audio file.

        args:
            audio_path: path to audio file

        returns:
            list of diarization segments sorted by start time
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        self.load_model()

        console.print(f"[cyan]diarizing {audio_path.name}...[/cyan]")
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("identifying speakers...", total=None)

            # load and resample audio to expected sample rate (16kHz mono)
            import soundfile as sf

            audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            audio = audio[:, 0]  # mono

            if sr != self.sd.sample_rate:
                import librosa

                progress.update(task, description="resampling audio...")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sd.sample_rate)

            progress.update(task, description="identifying speakers...")
            result = self.sd.process(audio).sort_by_start_time()

        duration = time.time() - start_time
        audio_duration = len(audio) / self.sd.sample_rate

        segments = [DiarizationSegment(start=r.start, end=r.end, speaker=r.speaker) for r in result]

        # post-clustering cleanup: merge small clusters into nearest large speaker.
        # only when auto-detecting speakers (num_speakers=-1) and enough segments
        # to make the heuristic meaningful.
        if self.num_speakers < 0 and len(segments) >= MIN_TOTAL_SEGMENTS_FOR_MERGE:
            min_segs = max(MIN_CLUSTER_FLOOR, int(len(segments) * MIN_CLUSTER_FRACTION))
            segments = _merge_small_clusters(segments, min_segs)

        num_speakers = len({s.speaker for s in segments})

        console.print(
            f"[green]✓[/green] diarization complete: "
            f"{num_speakers} speakers, {len(segments)} segments "
            f"in {duration:.1f}s "
            f"(speed: {audio_duration / duration:.1f}x)"
            if audio_duration > 0
            else ""
        )

        return segments

    def unload_model(self) -> None:
        """unload diarization models."""
        if hasattr(self, "sd") and self.sd is not None:
            del self.sd
        self.sd = None
        import gc

        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, Exception):
            pass
        console.print("[dim]diarization models unloaded[/dim]")


def _merge_small_clusters(
    segments: list[DiarizationSegment],
    min_segments: int,
) -> list[DiarizationSegment]:
    """
    merge speaker clusters with fewer than min_segments into the nearest
    large-cluster speaker by temporal proximity.

    this mimics pyannote 3.1's min_cluster_size: spurious micro-clusters
    caused by noise, voice variation, or embedding instability get absorbed
    into the dominant speakers.

    args:
        segments: diarization segments from sherpa-onnx
        min_segments: minimum segments a speaker must have to survive

    returns:
        segments with small-cluster speakers reassigned
    """
    if not segments:
        return segments

    counts = Counter(s.speaker for s in segments)
    large_speakers = {spk for spk, cnt in counts.items() if cnt >= min_segments}

    if not large_speakers or large_speakers == set(counts.keys()):
        # no small clusters, or everything is small (don't merge away all speakers)
        return segments

    small_speakers = set(counts.keys()) - large_speakers
    if not small_speakers:
        return segments

    merged = 0
    for seg in segments:
        if seg.speaker in small_speakers:
            # find the nearest large-cluster segment by temporal midpoint distance
            seg_mid = (seg.start + seg.end) / 2
            best_speaker = None
            best_dist = float("inf")
            for other in segments:
                if other.speaker in large_speakers:
                    other_mid = (other.start + other.end) / 2
                    dist = abs(seg_mid - other_mid)
                    if dist < best_dist:
                        best_dist = dist
                        best_speaker = other.speaker
            if best_speaker is not None:
                seg.speaker = best_speaker
                merged += 1

    if merged > 0:
        final_count = len({s.speaker for s in segments})
        console.print(
            f"[dim]merged {merged} segment(s) from "
            f"{len(small_speakers)} small cluster(s) "
            f"→ {final_count} speakers[/dim]"
        )

    return segments


def assign_speakers(
    transcript_segments: list,
    diarization_segments: list[DiarizationSegment],
) -> tuple[list, list[dict]]:
    """
    assign speaker IDs to transcript segments based on time overlap.

    uses maximum temporal overlap to deterministically assign each transcript
    segment to the speaker whose diarization segment overlaps the most.

    args:
        transcript_segments: list of TranscriptSegment objects
        diarization_segments: list of DiarizationSegment objects

    returns:
        tuple of (updated segments, speakers metadata list)
    """
    if not diarization_segments:
        return transcript_segments, []

    # assign speaker with maximum overlap to each transcript segment
    for seg in transcript_segments:
        best_speaker = None
        best_overlap = 0.0
        for diar in diarization_segments:
            overlap = max(0, min(seg.end, diar.end) - max(seg.start, diar.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar.speaker
        seg.speaker_id = best_speaker

    # remap non-sequential speaker IDs to 0-indexed
    # (sherpa-onnx may return e.g. speaker_00, speaker_03, speaker_04)
    raw_ids = sorted({s.speaker_id for s in transcript_segments if s.speaker_id is not None})
    id_map = {old: new for new, old in enumerate(raw_ids)}
    for seg in transcript_segments:
        if seg.speaker_id is not None:
            seg.speaker_id = id_map[seg.speaker_id]

    # build speaker metadata
    unique_speakers = sorted(
        {s.speaker_id for s in transcript_segments if s.speaker_id is not None}
    )
    speakers = [{"id": s, "label": f"Speaker {s + 1}"} for s in unique_speakers]

    return transcript_segments, speakers
