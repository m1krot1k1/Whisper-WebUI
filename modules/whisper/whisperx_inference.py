import os
import time
import numpy as np
import torch
from typing import BinaryIO, Union, Tuple, List, Callable
import whisperx
import gradio as gr

from modules.utils.paths import (WHISPERX_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline


class WhisperXInference(BaseTranscriptionPipeline):
    def __init__(self,
                 model_dir: str = WHISPERX_MODELS_DIR,
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
        self.device = self.get_device()
        self.available_models = self.model_paths.keys()

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
        """
        transcribe method for whisperX.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.
        *whisper_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        -------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()

        params = WhisperParams.from_list(list(whisper_params))

        if params.model_size != self.current_model_size or self.model is None:
            self.update_model(params.model_size, progress)

        # Load audio if it's a file path
        if isinstance(audio, str):
            import torchaudio
            audio_data, sample_rate = torchaudio.load(audio)
            audio = audio_data.numpy()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)  # Convert to mono if stereo

        progress(0, desc="Transcribing with WhisperX..")

        # Transcribe with WhisperX
        result = self.model.transcribe(
            audio,
            language=params.lang if params.lang != "auto" else None,
            batch_size=16,  # Default batch size
            print_progress=True
        )

        # Align the output
        if result["segments"]:
            progress(0.5, desc="Aligning transcription..")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )

        segments_result = []
        for segment in result["segments"]:
            segments_result.append(Segment(
                text=segment["text"],
                start=segment["start"],
                end=segment["end"],
                words=[
                    Word(
                        start=word["start"],
                        end=word["end"],
                        word=word["word"],
                        probability=word.get("probability", 0.0)
                    ) for word in segment.get("words", [])
                ] if "words" in segment else None
            ))

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """
        Update current model setting

        Parameters
        ----------
        model_size: str
            Size of whisper model.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing WhisperX Model..")

        self.current_model_size = model_size

        # Use CUDA if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = whisperx.load_model(
            model_size,
            device=device,
            compute_type="float16" if device == "cuda" else "float32",
            download_root=self.model_dir
        )

    def get_model_paths(self):
        """
        Get available models from models path including fine-tuned model.

        Returns
        -------
        Name list of models
        """
        # WhisperX uses standard Whisper model names
        import whisper
        model_paths = {model: model for model in whisper.available_models()}

        # Add any custom models in the directory
        existing_models = os.listdir(self.model_dir)
        wrong_dirs = [".locks", "whisperx_models_will_be_saved_here"]
        existing_models = list(set(existing_models) - set(wrong_dirs))

        for model_name in existing_models:
            if model_name not in whisper.available_models():
                model_paths[model_name] = os.path.join(self.model_dir, model_name)

        return model_paths

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
