import os
import time
import huggingface_hub
import numpy as np
import torch
from typing import BinaryIO, Union, Tuple, List, Callable
import faster_whisper
from faster_whisper.vad import VadOptions
import ast
import ctranslate2
import whisper
import gradio as gr
from argparse import Namespace
import librosa

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline


class FasterWhisperInference(BaseTranscriptionPipeline):
    def __init__(self,
                 model_dir: str = FASTER_WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        super().__init__(
            model_dir=model_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
            output_dir=output_dir
        )
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_paths = self.get_model_paths()
        self.device = BaseTranscriptionPipeline.get_device()
        self.available_models = self.model_paths.keys()

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   params: Optional[WhisperParams] = None,
                   ) -> Tuple[List[Segment], float]:
        """
        transcribe method for faster-whisper.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.
        params: WhisperParams
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()

        if params is None:
            params = WhisperParams()

        if params.model_size != self.current_model_size or self.model is None or self.current_compute_type != params.compute_type:
            self.update_model(params.model_size, params.compute_type, progress)

        # Use Silero STT if enabled
        if hasattr(self, 'use_silero_stt') and self.use_silero_stt:
            if isinstance(audio, str):
                audio_np, _ = librosa.load(audio, sr=16000)
            elif isinstance(audio, bytes):
                audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32767.0
            else:
                audio_np = audio

            silero_text = self.transcribe_with_silero(audio_np, params.lang or 'ru')
            if silero_text:
                # Create a single segment with Silero result
                segment = Segment(
                    text=silero_text,
                    start=0.0,
                    end=10.0  # approximate
                )
                return [segment], time.time() - start_time

        segments, info = self.model.transcribe(
            audio=audio,
            language=params.lang,
            task="translate" if params.is_translate else "transcribe",
            beam_size=params.beam_size,
            log_prob_threshold=params.log_prob_threshold,
            no_speech_threshold=params.no_speech_threshold,
            best_of=params.best_of,
            patience=params.patience,
            temperature=params.temperature,
            initial_prompt=params.initial_prompt,
            compression_ratio_threshold=params.compression_ratio_threshold,
            length_penalty=params.length_penalty,
            repetition_penalty=params.repetition_penalty,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            prefix=params.prefix,
            suppress_blank=params.suppress_blank,
            suppress_tokens=ast.literal_eval(params.suppress_tokens) if isinstance(params.suppress_tokens, str) else params.suppress_tokens,
            max_initial_timestamp=params.max_initial_timestamp,
            word_timestamps=True,  # Set it to always True as it reduces hallucinations
            prepend_punctuations=params.prepend_punctuations or "\"'“¿([{-",
            append_punctuations=params.append_punctuations or "\"'.。,，!！?？:：”)]}、",
            max_new_tokens=params.max_new_tokens,
            chunk_length=params.chunk_length,
            hallucination_silence_threshold=params.hallucination_silence_threshold,
            hotwords=params.hotwords,
            language_detection_threshold=params.language_detection_threshold,
            language_detection_segments=params.language_detection_segments,
            prompt_reset_on_temperature=params.prompt_reset_on_temperature,
        )
        progress(0, desc="Loading audio..")

        segments_result = []
        if info and hasattr(info, 'duration'):
            for segment in segments:
                progress_n = segment.start / info.duration
                progress(progress_n, desc="Transcribing..")
                if progress_callback is not None:
                    progress_callback(progress_n)
                segments_result.append(Segment.from_faster_whisper(segment))
        else:
            segments_result = [Segment.from_faster_whisper(s) for s in segments]

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """
        Update current model setting

        Parameters
        ----------
        model_size: str
            Size of whisper model. If you enter the huggingface repo id, it will try to download the model
            automatically from huggingface.
        compute_type: str
            Compute type for transcription.
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing Model..")

        model_size_dirname = model_size.replace("/", "--") if "/" in model_size else model_size
        model_path = os.path.join(self.model_dir, model_size_dirname)

        if not os.path.exists(model_path) or not os.listdir(model_path):
            progress(0, desc=f"Downloading {model_size} model..")
            try:
                model_path = huggingface_hub.snapshot_download(
                    repo_id=model_size,
                    cache_dir=self.model_dir,
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                raise gr.Error(f"Failed to download model: {e}")

        self.model = faster_whisper.WhisperModel(
            model_size_or_path=model_path,
            device=self.device,
            compute_type=compute_type,
        )
        self.current_model_size = model_size
        self.current_compute_type = compute_type

    def get_model_paths(self) -> dict:
        model_paths = {}
        for root, dirs, files in os.walk(self.model_dir):
            for file in files:
                if file == "config.json":
                    model_name = os.path.basename(root)
                    model_paths[model_name] = root
        return model_paths

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def format_suppress_tokens_str(suppress_tokens_str: str) -> List[int]:
        try:
            suppress_tokens = ast.literal_eval(suppress_tokens_str)
            if not isinstance(suppress_tokens, list) or not all(isinstance(item, int) for item in suppress_tokens):
                raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
            return suppress_tokens
        except Exception as e:
            raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
