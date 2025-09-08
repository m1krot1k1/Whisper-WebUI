import functools
import uuid
import numpy as np
from fastapi import (
    File,
    UploadFile,
)
import gradio as gr
from fastapi import APIRouter, BackgroundTasks, Depends, Response, status
from typing import List, Dict
from sqlalchemy.orm import Session
from datetime import datetime
from modules.whisper.data_classes import *
from modules.utils.paths import BACKEND_CACHE_DIR
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from backend.common.audio import read_audio
from backend.common.models import QueueResponse
from backend.common.config_loader import load_server_config
from backend.db.task.dao import (
    add_task_to_db,
    get_db_session,
    update_task_status_in_db
)
from backend.db.task.models import TaskStatus, TaskType

transcription_router = APIRouter(prefix="/transcription", tags=["Transcription"])


def create_progress_callback(identifier: str):
    def progress_callback(progress_value: float):
        update_task_status_in_db(
            identifier=identifier,
            update_data={
                "uuid": identifier,
                "status": TaskStatus.IN_PROGRESS,
                "progress": round(progress_value, 2),
                "updated_at": datetime.utcnow()
            },
        )
    return progress_callback


@functools.lru_cache
def get_pipeline() -> 'FasterWhisperInference':
    config = load_server_config()["whisper"]
    inferencer = FasterWhisperInference(
        output_dir=BACKEND_CACHE_DIR
    )
    inferencer.update_model(
        model_size=config["model_size"],
        compute_type=config["compute_type"]
    )
    # Set default parameters from config
    inferencer.default_beam_size = config.get("beam_size", 5)
    inferencer.default_temperature = config.get("temperature", 0.0)
    inferencer.enable_preprocessing = config.get("enable_preprocessing", True)
    inferencer.enable_postprocessing = config.get("enable_postprocessing", True)
    inferencer.use_silero_stt = config.get("use_silero_stt", False)
    return inferencer


def run_transcription(
    audio: np.ndarray,
    params: TranscriptionPipelineParams,
    identifier: str,
) -> List[Segment]:
    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.IN_PROGRESS,
            "updated_at": datetime.utcnow()
        },
    )

    # Set defaults from config
    config = load_server_config()["whisper"]
    if params.whisper.beam_size == 5:  # default
        params.whisper.beam_size = config.get("beam_size", 5)
    if params.whisper.temperature == 0.0:  # default
        params.whisper.temperature = config.get("temperature", 0.0)

    # Apply chunk_length default if caller left the default value
    try:
        default_chunk = params.__fields__["whisper"].type_.__fields__["chunk_length"].default
    except Exception:
        default_chunk = None
    if default_chunk is not None and params.whisper.chunk_length == default_chunk:
        params.whisper.chunk_length = config.get("chunk_length", params.whisper.chunk_length)

    # Apply VAD default from config if caller left default
    if params.vad.vad_filter is False:
        params.vad.vad_filter = config.get("vad_filter", params.vad.vad_filter)

    # Apply BGM separation default
    if params.bgm_separation.is_separate_bgm is False:
        params.bgm_separation.is_separate_bgm = config.get("separate_bgm_by_default", params.bgm_separation.is_separate_bgm)

    # Ensure pipeline flags reflect server config (in case cached pipeline was created earlier)
    pipeline = get_pipeline()
    pipeline.enable_preprocessing = config.get("enable_preprocessing", pipeline.enable_preprocessing)
    pipeline.enable_postprocessing = config.get("enable_postprocessing", pipeline.enable_postprocessing)
    pipeline.use_silero_stt = config.get("use_silero_stt", pipeline.use_silero_stt)

    # If diarization is requested, auto-disable BGM separation to avoid timing/artifact issues
    if params.diarization.is_diarize:
        if params.bgm_separation.is_separate_bgm:
            print(f"[transcription] Auto-disabling BGM separation for task {identifier} because diarization is enabled")
        params.bgm_separation.is_separate_bgm = False

    # Log applied params for traceability (stored in task params and printed)
    # Runtime guard: avoid known unstable decoding combos (beam large + very small chunk)
    guard_adjustments = {}
    try:
        # original unstable combo guard
        if params.whisper.beam_size >= 8 and params.whisper.chunk_length <= 20:
            old_beam = params.whisper.beam_size
            old_chunk = params.whisper.chunk_length
            params.whisper.beam_size = 5
            params.whisper.chunk_length = max(params.whisper.chunk_length, 30)
            guard_adjustments.update({"old_beam": old_beam, "old_chunk": old_chunk, "new_beam": params.whisper.beam_size, "new_chunk": params.whisper.chunk_length})
            print(f"[transcription] Guard: adjusted unstable params for task {identifier}: {guard_adjustments}")

        # Additional safe defaults when diarization is requested: keep beam moderate and chunk reasonably large
        if params.diarization.is_diarize:
            dia_adjust = {}
            if params.whisper.beam_size > 5:
                dia_adjust['old_beam'] = params.whisper.beam_size
                params.whisper.beam_size = 5
                dia_adjust['new_beam'] = params.whisper.beam_size
            if params.whisper.chunk_length < 30:
                dia_adjust['old_chunk'] = params.whisper.chunk_length
                params.whisper.chunk_length = 30
                dia_adjust['new_chunk'] = params.whisper.chunk_length
            if dia_adjust:
                guard_adjustments.setdefault('diarize_adjustments', dia_adjust)
                print(f"[transcription] Diarize-safe adjustments for task {identifier}: {dia_adjust}")
    except Exception:
        guard_adjustments = {}

    print(f"[transcription] Applied params for task {identifier}: beam={params.whisper.beam_size}, chunk={params.whisper.chunk_length}, vad={params.vad.vad_filter}, bgm={params.bgm_separation.is_separate_bgm}, preproc={pipeline.enable_preprocessing}, postproc={pipeline.enable_postprocessing}")

    # Persist the effective (applied) params into task DB so the UI can show them
    try:
        update_task_status_in_db(
            identifier=identifier,
            update_data={
                "uuid": identifier,
                "applied_params": params.to_dict(),
                "guard_adjustments": guard_adjustments,
                "updated_at": datetime.utcnow(),
            },
        )
    except Exception as e:
        print(f"[transcription] Warning: failed to write applied params to DB for task {identifier}: {e}")

    progress_callback = create_progress_callback(identifier)
    segments, elapsed_time = pipeline.run(
        audio,
        gr.Progress(),
        "SRT",
        False,
        progress_callback,
        *params.to_list()
    )
    segments = [seg.model_dump() for seg in segments]

    # Persist generated SRT to outputs and record path
    try:
        from modules.utils.subtitle_manager import generate_file
        from modules.utils.files_manager import safe_filename

        file_name = safe_filename(f"transcription_{identifier}")
        _, file_path = generate_file(
            output_format="srt",
            output_dir=pipeline.output_dir,
            result={"segments": segments},
            output_file_name=file_name,
            add_timestamp=False,
        )
        print(f"[transcription] Generated SRT for task {identifier}: {file_path}")
    except Exception as e:
        file_path = None
        print(f"[transcription] Failed to persist SRT for task {identifier}: {e}")

    update_task_status_in_db(
        identifier=identifier,
        update_data={
            "uuid": identifier,
            "status": TaskStatus.COMPLETED,
            "result": segments,
            "srt_path": file_path,
            "updated_at": datetime.utcnow(),
            "duration": elapsed_time,
            "progress": 1.0,
        },
    )
    return segments


@transcription_router.post(
    "/",
    response_model=QueueResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Transcribe Audio",
    description="Process the provided audio or video file to generate a transcription.",
)
async def transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    whisper_params: WhisperParams = Depends(),
    vad_params: VadParams = Depends(),
    bgm_separation_params: BGMSeparationParams = Depends(),
    diarization_params: DiarizationParams = Depends(),
) -> QueueResponse:
    if not isinstance(file, np.ndarray):
        audio, info = await read_audio(file=file)
    else:
        audio, info = file, None

    params = TranscriptionPipelineParams(
        whisper=whisper_params,
        vad=vad_params,
        bgm_separation=bgm_separation_params,
        diarization=diarization_params
    )

    identifier = add_task_to_db(
        status=TaskStatus.QUEUED,
        file_name=file.filename,
        audio_duration=info.duration if info else None,
        language=params.whisper.lang,
        task_type=TaskType.TRANSCRIPTION,
        task_params=params.to_dict(),
    )

    background_tasks.add_task(
        run_transcription,
        audio=audio,
        params=params,
        identifier=identifier,
    )

    return QueueResponse(identifier=identifier, status=TaskStatus.QUEUED, message="Transcription task has queued")


