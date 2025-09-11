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
from modules.utils.logger import get_logger

logger = get_logger()


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

        # Load Whisper configuration (WhisperX uses main whisper config)
        import yaml
        from modules.utils.paths import SERVER_CONFIG_PATH
        config = yaml.safe_load(open(SERVER_CONFIG_PATH, 'r'))
        self.whisper_config = config.get('whisper', {})

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
            self.update_model(params.model_size, params, progress)

        # Load audio if it's a file path
        if isinstance(audio, str):
            import torchaudio
            audio_data, sample_rate = torchaudio.load(audio)
            audio = audio_data.numpy()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)  # Convert to mono if stereo
            
            # Check if audio is empty
            if audio.size == 0:
                logger.warning(f"Audio file {audio} appears to be empty or corrupted.")
                return [Segment()], 0

        progress(0, desc="Transcribing with WhisperX..")

        # Transcribe with WhisperX using only basic supported parameters
        result = self.model.transcribe(
            audio,
            language=params.lang if params.lang != "auto" else None,
            batch_size=params.batch_size
        )

        # Align the output for better word spacing (if enabled)
        enable_alignment = self.whisper_config.get('enable_alignment', True)
        if result["segments"] and enable_alignment:
            progress(0.5, desc="Aligning transcription..")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"],
                    device=self.device
                )

                return_char_alignments = self.whisper_config.get('return_char_alignments', True)

                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=return_char_alignments
                )
            except Exception as e:
                print(f"Warning: Alignment failed, using original transcription: {e}")
                # Continue with original result if alignment fails

        segments_result = []
        for i, segment in enumerate(result["segments"]):
            # Get the original text from the segment
            original_text = segment["text"]
            
            # If we have word-level information after alignment, reconstruct text with proper spacing
            if "words" in segment and segment["words"] and len(segment["words"]) > 0:
                # Reconstruct text from words to ensure proper spacing
                reconstructed_text = " ".join([word["word"] for word in segment["words"]])
                
                # Use reconstructed text if it's different from original (better spacing)
                # This handles cases where original text has spacing issues
                if reconstructed_text.strip() != original_text.strip():
                    processed_text = reconstructed_text
                else:
                    processed_text = original_text
            else:
                # No word-level info available, use original text
                processed_text = original_text
            
            # Create word list if available
            words = None
            if "words" in segment and segment["words"]:
                words = [
                    Word(
                        start=word["start"],
                        end=word["end"],
                        word=word["word"],
                        probability=word.get("probability", 0.0)
                    ) for word in segment["words"]
                ]

            segments_result.append(Segment(
                text=processed_text,
                start=segment["start"],
                end=segment["end"],
                words=words
            ))

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     params,
                     progress: gr.Progress = gr.Progress()
                     ):
        """
        Update current model setting

        Parameters
        ----------
        model_size: str
            Size of whisper model.
        params: WhisperParams
            Parameters object containing language and other settings.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing WhisperX Model..")

        self.current_model_size = model_size

        # Use device from config or auto-detect
        device = self.whisper_config.get('device', "cuda" if torch.cuda.is_available() else "cpu")

        # Load WhisperX model with proper parameters for word spacing
        self.model = whisperx.load_model(
            model_size,
            device=device,
            compute_type=self.whisper_config.get('compute_type', "float16" if device == "cuda" else "float32"),
            download_root=self.model_dir,
            # Additional parameters for better word recognition
            language=params.lang if hasattr(params, 'lang') and params.lang and params.lang != "auto" else None
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
