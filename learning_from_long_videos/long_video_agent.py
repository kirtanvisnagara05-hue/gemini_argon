"""Long-Horizon YouTube Gameplay Knowledge Extractor powered by Gemini 4 Argon.

Workflow:
1. Accepts a YouTube URL (e.g., a 45-minute SimCity BuildIt gameplay video) and
   downloads it as an MP4 without audio using `yt-dlp` + `imageio-ffmpeg`.
2. Uploads the video to Gemini Files API and calls `client.models.count_tokens`
   with `gemini-4-argon` to compute the exact token count before running inference.
3. Displays the video duration, file size, exact token count, and percentage of
   Argon's 1,000,000-token context window, then asks for your approval:
   - Approve (`y` / `yes` / Enter): Sends the entire video in a single pass to
     `gemini-4-argon`.
   - Divide (`divide` / `d`): Automatically splits the video into chunks sized to
     fit your target token budget (or number of parts), processes each chunk
     with Argon, and saves all decomposed tasks.
4. Decomposes all tasks performed in the gameplay video. Each task includes:
   - `task_name`: Name of the task
   - `when_to_do_it`: Strategic reasoning & game conditions for when/why to
     perform this task (e.g., building a park specifically adjacent to clustered
     residential zones that yield maximum population boost, or upgrading water
     before zoning new residences when capacity is near demand).
   - `steps`: Ordered atomic steps (`action`, `who`, `briefing`).
5. Saves each task as its own JSON file in `learning_from_long_videos/knowledge/<task_name>.json`
   and logs all token & timing metrics to `learning_from_long_videos/performance_logs.json`
   (and updates `Learning_agent/performance_argon.json`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
from dotenv import load_dotenv
from google import genai


LONG_VIDEO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LONG_VIDEO_DIR.parent
LEARNING_AGENT_DIR = PROJECT_ROOT / "Learning_agent"
ENV_FILE = LONG_VIDEO_DIR / ".env"
FALLBACK_ENV_FILE = LEARNING_AGENT_DIR / ".env"

DOWNLOADS_DIR = LONG_VIDEO_DIR / "downloads"
CHUNKS_DIR = LONG_VIDEO_DIR / "chunks"
KNOWLEDGE_DIR = LONG_VIDEO_DIR / "knowledge"
PERFORMANCE_LOGS_FILE = LONG_VIDEO_DIR / "performance_logs.json"
SHARED_PERFORMANCE_FILE = LEARNING_AGENT_DIR / "performance_argon.json"

DEFAULT_MODEL = "gemini-4-argon"
MAX_CONTEXT_TOKENS = 1_000_000
ALLOWED_ACTIONS = {"click", "drag", "find", "wait"}

SYSTEM_PROMPT = """You are a Master SimCity BuildIt Strategist and Long-Video Workflow Decomposer powered by Gemini 4 Argon.

You are watching a long SimCity BuildIt gameplay video. Your job is to:
1. Identify and decompose EVERY distinct task performed by the player across the video in chronological order (for example: building road connections, zoning residential areas, upgrading residential buildings, constructing factories, producing materials like metal/wood, placing city services like water/power/fire, placing specialization buildings like parks to maximize population boost, collecting taxes/simoleons, expanding city storage, etc.).
2. For each task, deeply analyze and include `"when_to_do_it"`:
   - Explain the exact strategic trigger, precondition, and spatial/economic reasoning for WHY and WHEN the agent should perform this task based on what is shown in the gameplay.
   - Example: If the player builds a park next to residential buildings, do NOT just say "when you want a park." Reason about the gameplay context: e.g., "Perform this when residential buildings are clustered and upgraded, placing the park where its green coverage area overlaps the maximum number of residential zones to achieve the highest population boost and unlock next-tier unlocks."
   - Include resource preconditions, UI indicators (red/yellow alert icons, capacity vs demand bars, hard-hat bubbles), and placement strategy.
3. For each task, generate the exact executor-ready `"steps"` describing what the player did:
   - `"action"`: must be one of `"click"`, `"drag"`, `"find"`, or `"wait"`.
   - `"who"`: the specific visible UI icon, button, building, material, or target on screen.
   - `"briefing"`: clear visual intent and destination guidance for a vision-guided ADB executor.

Return ONLY valid JSON matching this exact schema (no markdown fences, no commentary):
{
    "video_summary": "Brief overview of the city progression shown in this video",
    "tasks": [
        {
            "task_name": "Build Park for Maximum Residential Population Boost",
            "timestamp_range": "MM:SS - MM:SS",
            "when_to_do_it": {
                "trigger_condition": "When multiple Residential Zones are built and clustered along roads and you want to maximize population growth.",
                "strategic_reasoning": "In the gameplay, the park is specifically placed adjacent to the densest cluster of residential buildings rather than an empty road so its green influence radius covers the maximum number of homes, giving the highest population push.",
                "preconditions": [
                    "Parks specialization is unlocked (City Level 4+)",
                    "Sufficient Simoleons or SimCash available",
                    "Clustered residential buildings connected to a road"
                ],
                "placement_rule": "Drag the park preview until the green coverage highlight overlaps the maximum number of existing residential buildings and shows the highest +Population indicator."
            },
            "steps": [
                {
                    "action": "click",
                    "who": "Services / Specialization icon in bottom-right menu",
                    "briefing": "open the building specialization menu."
                },
                {
                    "action": "click",
                    "who": "Parks park-bench icon",
                    "briefing": "open the Parks catalog to view available parks."
                },
                {
                    "action": "drag",
                    "who": "Small Fountain Park card",
                    "briefing": "drag the park onto the road spot directly beside the densest residential cluster to maximize the green population boost indicator."
                },
                {
                    "action": "click",
                    "who": "green checkmark",
                    "briefing": "confirm park placement and apply the population boost to surrounding homes."
                }
            ]
        }
    ]
}
"""


@dataclass
class VideoMetadata:
    path: Path
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    size_mb: float


def build_genai_client() -> tuple[genai.Client, str, str]:
    load_dotenv(ENV_FILE)
    load_dotenv(FALLBACK_ENV_FILE)
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        api_key = getpass("Enter your Gemini API key: ").strip()
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY is missing. Add it to {ENV_FILE} or {FALLBACK_ENV_FILE}.")

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "medium").strip() or "medium"
    return genai.Client(api_key=api_key), model_name, thinking_level


def inspect_video(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)
    return VideoMetadata(
        path=video_path,
        duration_seconds=round(duration, 2),
        fps=round(fps, 2),
        frame_count=frame_count,
        width=width,
        height=height,
        size_mb=size_mb,
    )


def download_youtube_video(youtube_url: str, max_height: int = 480) -> tuple[Path, float]:
    """Download video-only MP4 from YouTube using yt-dlp + bundled ffmpeg."""
    import yt_dlp

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    print(f"\n[YouTube] Downloading video from: {youtube_url}")
    print(f"[YouTube] Target resolution: <= {max_height}p (video-only, no audio for faster upload)...")

    t0 = time.monotonic()
    out_tmpl = str(DOWNLOADS_DIR / "%(id)s_%(height)sp.%(ext)s")
    ydl_opts = {
        "format": f"bestvideo[height<={max_height}][ext=mp4]/best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "outtmpl": out_tmpl,
        "ffmpeg_location": ffmpeg_exe,
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
        "quiet": False,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))
        if not downloaded_path.exists():
            mp4_candidate = downloaded_path.with_suffix(".mp4")
            if mp4_candidate.exists():
                downloaded_path = mp4_candidate

    download_seconds = round(time.monotonic() - t0, 2)
    print(f"[YouTube] Downloaded to: {downloaded_path} in {download_seconds}s")
    return downloaded_path, download_seconds


def upload_and_wait(client: genai.Client, video_path: Path) -> tuple[Any, float]:
    """Upload MP4 to Gemini Files API and wait until state is ACTIVE."""
    t0 = time.monotonic()
    print(f"\n[Argon] Uploading '{video_path.name}' ({video_path.stat().st_size / (1024 * 1024):.2f} MB) to Gemini Files API...")
    uploaded = client.files.upload(
        file=video_path,
        config={"mime_type": "video/mp4", "display_name": video_path.name},
    )
    file_name = getattr(uploaded, "name", None)
    if file_name:
        while True:
            state = str(getattr(uploaded, "state", "") or "").upper()
            if "ACTIVE" in state or not state:
                break
            if "FAILED" in state:
                raise RuntimeError(f"Gemini Files API failed processing {video_path.name} (state={state}).")
            elapsed = int(time.monotonic() - t0)
            print(f"\r[Argon] Processing video on Gemini servers ({state}, {elapsed}s elapsed)...", end="", flush=True)
            time.sleep(3.0)
            uploaded = client.files.get(name=file_name)
    upload_seconds = round(time.monotonic() - t0, 2)
    print(f"\n[Argon] Video ACTIVE on Gemini in {upload_seconds}s -> {uploaded.uri}")
    return uploaded, upload_seconds


def count_video_tokens(
    client: genai.Client,
    model_name: str,
    uploaded_file: Any,
    prompt_text: str,
) -> tuple[int, float]:
    """Count exact tokens for the uploaded video + system/user prompt using Gemini's count_tokens API."""
    t0 = time.monotonic()
    print(f"[Argon] Counting exact tokens for '{model_name}'...")
    resp = client.models.count_tokens(
        model=model_name,
        contents=[uploaded_file, SYSTEM_PROMPT + "\n\n" + prompt_text],
    )
    count_sec = round(time.monotonic() - t0, 2)
    total_tokens = int(getattr(resp, "total_tokens", 0) or 0)
    return total_tokens, count_sec


def split_video_into_chunks(video_path: Path, num_chunks: int) -> list[Path]:
    """Split a long MP4 into `num_chunks` equal-duration segments using ffmpeg stream copy."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    meta = inspect_video(video_path)
    total_duration = max(1.0, meta.duration_seconds)
    segment_seconds = math.ceil(total_duration / num_chunks)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    chunk_paths: list[Path] = []
    for idx in range(num_chunks):
        start_sec = idx * segment_seconds
        if start_sec >= total_duration:
            break
        out_chunk = CHUNKS_DIR / f"{video_path.stem}_part_{idx + 1:02d}_of_{num_chunks:02d}.mp4"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss",
            str(start_sec),
            "-i",
            str(video_path),
            "-t",
            str(segment_seconds),
            "-c",
            "copy",
            "-an",
            str(out_chunk),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        chunk_paths.append(out_chunk)
        print(
            f"  [Chunk {idx + 1}/{num_chunks}] {out_chunk.name} "
            f"({start_sec // 60:.0f}m{start_sec % 60:02.0f}s - {min(total_duration, start_sec + segment_seconds) // 60:.0f}m) "
            f"({out_chunk.stat().st_size / (1024 * 1024):.2f} MB)"
        )
    return chunk_paths


def _extract_usage_dict(usage_obj: Any) -> dict[str, Any]:
    if usage_obj is None:
        return {
            "total_input_tokens": 0,
            "input_tokens_by_modality": {},
            "total_thought_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_tokens": 0,
            "total_tokens": 0,
        }

    modality_map: dict[str, int] = {}
    for item in getattr(usage_obj, "input_tokens_by_modality", None) or []:
        modality = str(getattr(item, "modality", "UNKNOWN") or "UNKNOWN")
        tokens = int(getattr(item, "tokens", 0) or 0)
        modality_map[modality] = tokens

    return {
        "total_input_tokens": int(getattr(usage_obj, "total_input_tokens", 0) or 0),
        "input_tokens_by_modality": modality_map,
        "total_thought_tokens": int(getattr(usage_obj, "total_thought_tokens", 0) or 0),
        "total_output_tokens": int(getattr(usage_obj, "total_output_tokens", 0) or 0),
        "total_cached_tokens": int(getattr(usage_obj, "total_cached_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    elif not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _sanitize_filename(task_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\s]", "", task_name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "Unnamed_Task"


def normalize_and_save_tasks(
    parsed_response: dict[str, Any],
    source_label: str,
) -> list[dict[str, Any]]:
    """Validate extracted tasks and save each one to `knowledge/<task_name>.json`."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    raw_tasks = parsed_response.get("tasks")
    if not isinstance(raw_tasks, list):
        if "task_name" in parsed_response and "steps" in parsed_response:
            raw_tasks = [parsed_response]
        else:
            raise ValueError("Model response must contain a 'tasks' list.")

    saved_records: list[dict[str, Any]] = []
    for idx, task in enumerate(raw_tasks, start=1):
        if not isinstance(task, dict):
            continue
        task_name = str(task.get("task_name") or f"Task_{idx}").strip()
        when_to_do_it = task.get("when_to_do_it") or "Perform when this city objective is needed."
        timestamp_range = str(task.get("timestamp_range") or "").strip()
        raw_steps = task.get("steps") or []

        normalized_steps: list[dict[str, str]] = []
        for s in raw_steps:
            if not isinstance(s, dict):
                continue
            action = str(s.get("action", "click")).strip().lower()
            if action not in ALLOWED_ACTIONS:
                action = "click"
            who = str(s.get("who") or s.get("target") or "").strip()
            briefing = str(s.get("briefing") or s.get("description") or "").strip()
            if who and briefing:
                normalized_steps.append(
                    {
                        "action": action,
                        "who": who,
                        "briefing": briefing,
                    }
                )

        if not normalized_steps:
            continue

        task_doc = {
            "task_name": task_name,
            "timestamp_range": timestamp_range,
            "when_to_do_it": when_to_do_it,
            "steps": normalized_steps,
            "source_video": source_label,
        }

        safe_name = _sanitize_filename(task_name)
        file_path = KNOWLEDGE_DIR / f"{safe_name}.json"
        if file_path.exists():
            # Avoid overwriting if multiple tasks share a similar title
            counter = 2
            while (KNOWLEDGE_DIR / f"{safe_name}_{counter}.json").exists():
                counter += 1
            file_path = KNOWLEDGE_DIR / f"{safe_name}_{counter}.json"

        file_path.write_text(json.dumps(task_doc, indent=4), encoding="utf-8")
        saved_records.append(
            {
                "task_name": task_name,
                "file": str(file_path.relative_to(PROJECT_ROOT)),
                "step_count": len(normalized_steps),
                "timestamp_range": timestamp_range,
            }
        )
        print(f"  [Saved Knowledge] {file_path.name} ({len(normalized_steps)} steps)")

    return saved_records


def analyze_uploaded_video_with_argon(
    client: genai.Client,
    model_name: str,
    thinking_level: str,
    uploaded_file: Any,
    prompt_text: str,
    source_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], float, str | None]:
    """Send the uploaded video URI to `gemini-4-argon` and extract all tasks into `knowledge/`."""
    print(f"\n[Argon] Sending '{source_label}' to '{model_name}' (thinking_level={thinking_level})...")
    t0 = time.monotonic()
    response = client.interactions.create(
        model=model_name,
        system_instruction=SYSTEM_PROMPT,
        input=[
            {
                "type": "video",
                "uri": uploaded_file.uri,
                "mime_type": "video/mp4",
            },
            {
                "type": "text",
                "text": prompt_text,
            },
        ],
        generation_config={
            "thinking_level": thinking_level,
            "temperature": 0.2,
        },
    )
    inference_seconds = round(time.monotonic() - t0, 2)
    usage_dict = _extract_usage_dict(getattr(response, "usage", None))
    interaction_id = getattr(response, "id", None)

    raw_text = response.output_text
    if not raw_text:
        raise RuntimeError(f"Model '{model_name}' returned an empty response.")

    parsed = _extract_json_object(raw_text)
    saved_tasks = normalize_and_save_tasks(parsed, source_label=source_label)
    return saved_tasks, usage_dict, inference_seconds, interaction_id


def append_performance_log(entry: dict[str, Any], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history: dict[str, Any] = {"runs": []}
    if log_path.exists():
        try:
            loaded = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                history = loaded
        except Exception:
            pass

    history["runs"].append(entry)
    runs = history["runs"]
    history["summary"] = {
        "total_runs": len(runs),
        "total_tasks_extracted": sum(r.get("tasks_extracted_count", 0) for r in runs),
        "total_input_tokens": sum(r.get("tokens", {}).get("total_input_tokens", 0) for r in runs),
        "total_thought_tokens": sum(r.get("tokens", {}).get("total_thought_tokens", 0) for r in runs),
        "total_output_tokens": sum(r.get("tokens", {}).get("total_output_tokens", 0) for r in runs),
        "total_tokens": sum(r.get("tokens", {}).get("total_tokens", 0) for r in runs),
        "total_api_seconds": round(sum(r.get("timing_seconds", {}).get("total_api_seconds", 0.0) for r in runs), 2),
    }
    log_path.write_text(json.dumps(history, indent=4), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a 45-minute YouTube SimCity BuildIt gameplay video with Gemini 4 Argon and build a strategic task knowledge base."
    )
    parser.add_argument(
        "--url",
        default="",
        help="YouTube gameplay video URL (if omitted, you will be prompted to paste it).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional path to an already-downloaded local MP4 file (skips re-downloading).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=480,
        help="Maximum video resolution height to download from YouTube (default: 480p).",
    )
    parser.add_argument(
        "--target-tokens-per-chunk",
        type=int,
        default=250_000,
        help="Target token budget per chunk if you choose to divide the video (default: 250000).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip interactive approval prompt and process in one go if within context limits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client, model_name, thinking_level = build_genai_client()

    overall_start = time.monotonic()
    download_seconds = 0.0

    if args.video:
        video_path = args.video.expanduser().resolve()
        source_url = str(video_path)
    else:
        youtube_url = args.url.strip()
        if not youtube_url:
            youtube_url = input("Enter the YouTube gameplay URL: ").strip()
        if not youtube_url:
            raise RuntimeError("A YouTube URL is required.")
        source_url = youtube_url
        video_path, download_seconds = download_youtube_video(youtube_url, max_height=args.max_height)

    meta = inspect_video(video_path)
    print(
        f"\n=== Video Metadata ==="
        f"\nFile       : {meta.path.name}"
        f"\nDuration   : {meta.duration_seconds / 60:.2f} minutes ({meta.duration_seconds:.1f}s)"
        f"\nResolution : {meta.width}x{meta.height} @ {meta.fps} fps"
        f"\nFile Size  : {meta.size_mb:.2f} MB"
    )

    user_prompt = (
        "Analyze this entire SimCity BuildIt gameplay video. Decompose every task performed by the player, "
        "reason deeply about 'when_to_do_it' (strategic triggers, preconditions, and placement rules such as "
        "building parks around dense residential clusters for maximum population push), and extract the exact "
        "atomic steps for each task."
    )

    # Step 1: Upload full video and count exact tokens
    uploaded_full, upload_seconds = upload_and_wait(client, video_path)
    precounted_tokens = 0
    count_seconds = 0.0
    try:
        precounted_tokens, count_seconds = count_video_tokens(
            client=client,
            model_name=model_name,
            uploaded_file=uploaded_full,
            prompt_text=user_prompt,
        )
    except Exception as exc:
        print(f"[Argon] Warning: count_tokens returned an error ({exc}); estimating from duration (~260 tokens/sec).")
        precounted_tokens = int(meta.duration_seconds * 260)

    pct_of_context = (precounted_tokens / MAX_CONTEXT_TOKENS) * 100
    print(
        f"\n============================================================"
        f"\n  GEMINI 4 ARGON TOKEN COUNT REPORT (SINGLE-PASS CHECK)"
        f"\n============================================================"
        f"\n  Model                 : {model_name} (thinking_level={thinking_level})"
        f"\n  Video Duration        : {meta.duration_seconds / 60:.2f} mins ({meta.duration_seconds:.0f}s)"
        f"\n  Exact Input Tokens    : {precounted_tokens:,} tokens"
        f"\n  Argon Context Limit   : {MAX_CONTEXT_TOKENS:,} tokens"
        f"\n  Context Utilization   : {pct_of_context:.2f}%"
        f"\n============================================================"
    )

    if args.auto_approve:
        decision = "yes" if precounted_tokens <= MAX_CONTEXT_TOKENS else "divide"
    else:
        decision = (
            input(
                "\nHow would you like to proceed?\n"
                "  - Type 'yes' (or press Enter) to process the ENTIRE video in one single pass\n"
                "  - Type 'divide' (or 'd') to split the video into smaller token chunks\n"
                "  - Type 'no' (or 'q') to cancel\n"
                "Your choice [yes / divide / no]: "
            )
            .strip()
            .lower()
        )

    if decision in {"no", "n", "q", "quit", "cancel"}:
        print("[Cancelled] Cleaning up uploaded file on Gemini...")
        try:
            client.files.delete(name=uploaded_full.name)
        except Exception:
            pass
        return

    all_saved_tasks: list[dict[str, Any]] = []
    modality_totals: dict[str, int] = {}
    total_input_tokens = 0
    total_thought_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_inference_seconds = 0.0
    interaction_ids: list[str] = []
    mode_used = "single_pass"

    if decision in {"divide", "d", "split"}:
        mode_used = "divided_chunks"
        # Clean up the full uploaded file before uploading chunks
        try:
            client.files.delete(name=uploaded_full.name)
        except Exception:
            pass

        default_chunks = max(2, math.ceil(precounted_tokens / max(50_000, args.target_tokens_per_chunk)))
        if not args.auto_approve:
            custom_parts = input(
                f"Enter number of chunks to divide into (recommended: {default_chunks} chunks of ~{precounted_tokens // default_chunks:,} tokens each) [{default_chunks}]: "
            ).strip()
            num_chunks = int(custom_parts) if custom_parts.isdigit() and int(custom_parts) >= 2 else default_chunks
        else:
            num_chunks = default_chunks

        print(f"\n[Divider] Splitting {video_path.name} into {num_chunks} chunks...")
        chunk_files = split_video_into_chunks(video_path, num_chunks=num_chunks)

        for idx, chunk_path in enumerate(chunk_files, start=1):
            uploaded_chunk, chunk_up_sec = upload_and_wait(client, chunk_path)
            upload_seconds = round(upload_seconds + chunk_up_sec, 2)
            try:
                chunk_tokens, c_sec = count_video_tokens(client, model_name, uploaded_chunk, user_prompt)
                count_seconds = round(count_seconds + c_sec, 2)
                print(f"[Chunk {idx}/{len(chunk_files)}] Exact Token Count: {chunk_tokens:,} tokens")

                chunk_prompt = (
                    f"{user_prompt}\n\n"
                    f"Note: This is Part {idx} of {len(chunk_files)} of the gameplay video. "
                    "Extract all tasks shown in this segment."
                )
                saved, usage, inf_sec, int_id = analyze_uploaded_video_with_argon(
                    client=client,
                    model_name=model_name,
                    thinking_level=thinking_level,
                    uploaded_file=uploaded_chunk,
                    prompt_text=chunk_prompt,
                    source_label=chunk_path.name,
                )
                all_saved_tasks.extend(saved)
                total_inference_seconds = round(total_inference_seconds + inf_sec, 2)
                total_input_tokens += usage["total_input_tokens"]
                total_thought_tokens += usage["total_thought_tokens"]
                total_output_tokens += usage["total_output_tokens"]
                total_tokens += usage["total_tokens"]
                for k, v in usage["input_tokens_by_modality"].items():
                    modality_totals[k] = modality_totals.get(k, 0) + v
                if int_id:
                    interaction_ids.append(int_id)
            finally:
                try:
                    client.files.delete(name=uploaded_chunk.name)
                except Exception:
                    pass
    else:
        # Process the entire video in a single pass!
        try:
            saved, usage, inf_sec, int_id = analyze_uploaded_video_with_argon(
                client=client,
                model_name=model_name,
                thinking_level=thinking_level,
                uploaded_file=uploaded_full,
                prompt_text=user_prompt,
                source_label=video_path.name,
            )
            all_saved_tasks.extend(saved)
            total_inference_seconds = inf_sec
            total_input_tokens = usage["total_input_tokens"]
            total_thought_tokens = usage["total_thought_tokens"]
            total_output_tokens = usage["total_output_tokens"]
            total_tokens = usage["total_tokens"]
            modality_totals = usage["input_tokens_by_modality"]
            if int_id:
                interaction_ids.append(int_id)
        finally:
            try:
                client.files.delete(name=uploaded_full.name)
            except Exception:
                pass

    total_api_seconds = round(upload_seconds + count_seconds + total_inference_seconds, 2)
    total_elapsed_seconds = round(time.monotonic() - overall_start, 2)

    perf_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agent_type": "learning_from_long_videos",
        "mode": mode_used,
        "model": model_name,
        "thinking_level": thinking_level,
        "source_url": source_url,
        "video_file": video_path.name,
        "video_duration_minutes": round(meta.duration_seconds / 60, 2),
        "video_size_mb": meta.size_mb,
        "precounted_single_pass_tokens": precounted_tokens,
        "tasks_extracted_count": len(all_saved_tasks),
        "tasks_saved": all_saved_tasks,
        "interaction_ids": interaction_ids,
        "tokens": {
            "total_input_tokens": total_input_tokens,
            "input_tokens_by_modality": modality_totals,
            "total_thought_tokens": total_thought_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        },
        "timing_seconds": {
            "youtube_download_seconds": download_seconds,
            "upload_and_processing_seconds": upload_seconds,
            "token_counting_seconds": count_seconds,
            "model_inference_seconds": total_inference_seconds,
            "total_api_seconds": total_api_seconds,
            "total_elapsed_seconds": total_elapsed_seconds,
        },
    }

    append_performance_log(perf_entry, PERFORMANCE_LOGS_FILE)
    append_performance_log(perf_entry, SHARED_PERFORMANCE_FILE)

    print(
        f"\n=== Long-Video Argon Knowledge Extraction Complete ==="
        f"\nMode Used          : {mode_used}"
        f"\nTasks Extracted    : {len(all_saved_tasks)} saved in {KNOWLEDGE_DIR}"
        f"\nPre-counted Tokens : {precounted_tokens:,}"
        f"\nActual Input Tokens: {total_input_tokens:,} {modality_totals}"
        f"\nThought Tokens     : {total_thought_tokens:,}"
        f"\nOutput Tokens      : {total_output_tokens:,}"
        f"\nTotal Tokens       : {total_tokens:,}"
        f"\nDownload Time      : {download_seconds}s"
        f"\nUpload Time        : {upload_seconds}s"
        f"\nInference Time     : {total_inference_seconds}s"
        f"\nTotal API Time     : {total_api_seconds}s"
        f"\n[Saved] Knowledge files -> {KNOWLEDGE_DIR}"
        f"\n[Saved] Performance log -> {PERFORMANCE_LOGS_FILE}"
    )


if __name__ == "__main__":
    main()
