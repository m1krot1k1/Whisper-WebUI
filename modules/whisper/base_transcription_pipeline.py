import os
import whisper
from whisper import tokenizer
import ctranslate2
import gradio as gr
import torchaudio
from abc import ABC, abstractmethod
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import numpy as np
from datetime import datetime
from faster_whisper.vad import VadOptions
import gc
from copy import deepcopy
import time
import torch

from modules.uvr.music_separator import MusicSeparator
from modules.utils.paths import (WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR, CONFIGS_DIR)
from modules.utils.constants import *
from modules.utils.logger import get_logger
from modules.utils.subtitle_manager import *
from modules.utils.youtube_manager import get_ytdata, get_ytaudio
from modules.utils.files_manager import get_media_files, format_gradio_files, load_yaml, save_yaml, read_file
from modules.utils.audio_manager import validate_audio
from modules.whisper.data_classes import *
from modules.diarize.diarizer import Diarizer
from modules.vad.silero_vad import SileroVAD

# New imports for preprocessing
import librosa
from pydub import AudioSegment
from pydub.effects import normalize
from scipy.signal import butter, filtfilt


logger = get_logger()


class BaseTranscriptionPipeline(ABC):
    def __init__(self,
                 model_dir: str = WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.diarizer = Diarizer(
            model_dir=diarization_model_dir
        )
        self.vad = SileroVAD()
        self.music_separator = MusicSeparator(
            model_dir=uvr_model_dir,
            output_dir=os.path.join(output_dir, "UVR")
        )

        # Initialize preprocessing attributes
        self.enable_preprocessing = True
        self.enable_postprocessing = True
        self.use_silero_stt = False

        self.model = None
        self.current_model_size = None
        try:
            self.available_models = whisper.available_models()
        except AttributeError:
            self.available_models = []
        try:
            self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        except AttributeError:
            # Fallback list of common languages
            self.available_langs = ['en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su']
        self.device = BaseTranscriptionPipeline.get_device()
        self.available_compute_types = self.get_available_compute_type()
        self.current_compute_type = self.get_compute_type()

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Advanced audio preprocessing: noise reduction, equalization, normalization."""
        if not self.enable_preprocessing:
            return audio

        print("[preprocessing] Starting advanced audio preprocessing...")

        # Convert to pydub AudioSegment for processing
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

        # 1. High-pass filter to remove low-frequency noise
        print("[preprocessing] Applying high-pass filter...")
        # pydub's high_pass_filter is not available in all versions.
        # audio_segment = audio_segment.high_pass_filter(80)  # Remove frequencies below 80Hz

        # 2. Normalize audio levels
        print("[preprocessing] Normalizing audio levels...")
        audio_segment = normalize(audio_segment)

        # Convert back to numpy for advanced processing
        y = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32767

        # 3. Advanced noise reduction using spectral gating
        print("[preprocessing] Applying advanced noise reduction...")
        # Estimate noise profile from first 0.5 seconds (assuming it's relatively quiet)
        noise_samples = int(0.5 * sample_rate)
        if len(y) > noise_samples:
            noise_profile = y[:noise_samples]
            # Simple spectral gating
            y_denoised = self._spectral_gating(y, noise_profile, sample_rate)
        else:
            y_denoised = y

        # 4. Voice enhancement equalization
        print("[preprocessing] Applying voice enhancement equalization...")
        y_eq = self._voice_enhancement_eq(y_denoised, sample_rate)

        # 5. Dynamic range compression for better speech intelligibility
        print("[preprocessing] Applying dynamic range compression...")
        y_compressed = self._dynamic_compression(y_eq)

        # 6. Final normalization
        print("[preprocessing] Final normalization...")
        y_final = librosa.util.normalize(y_compressed)

        # Ensure max amplitude is 1.0
        y_final = y_final / np.max(np.abs(y_final))

        print("[preprocessing] Advanced preprocessing completed!")
        return y_final

    def _spectral_gating(self, audio: np.ndarray, noise_profile: np.ndarray, sample_rate: int) -> np.ndarray:
        """Advanced spectral gating for noise reduction."""
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

        # Compute noise profile in frequency domain
        noise_stft = librosa.stft(noise_profile, n_fft=n_fft, hop_length=hop_length)
        noise_magnitude = np.mean(np.abs(noise_stft), axis=1)

        # Spectral gating
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Compute SNR
        snr = magnitude / (noise_magnitude[:, np.newaxis] + 1e-8)

        # Apply gating with protection against division by zero
        gain = np.maximum(0, 1 - 1/(snr + 1e-8))
        gain = np.minimum(gain, 1.0)

        # Smooth gain over time
        gain_smooth = librosa.decompose.nn_filter(gain, aggregate=np.median, metric='cosine')

        # Apply gain
        magnitude_denoised = magnitude * gain_smooth

        # Reconstruct signal
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length)

        # Ensure output has the same length as input
        if len(audio_denoised) != len(audio):
            if len(audio_denoised) > len(audio):
                # Trim if longer
                audio_denoised = audio_denoised[:len(audio)]
            else:
                # Pad with zeros if shorter
                padding = len(audio) - len(audio_denoised)
                audio_denoised = np.pad(audio_denoised, (0, padding), mode='constant')

        return audio_denoised

    def _voice_enhancement_eq(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Voice enhancement equalization."""
        # Design voice enhancement filter
        # Boost frequencies important for speech intelligibility (2-5 kHz)
        # Cut frequencies that are less important or contain noise

        # Simple parametric EQ using scipy
        # High shelf boost for clarity (2-5 kHz)
        nyquist = sample_rate / 2
        high_freq = 2000 / nyquist
        b_high, a_high = butter(2, high_freq, btype='high')
        audio_high = filtfilt(b_high, a_high, audio)

        # Low shelf cut for rumble (< 100 Hz)
        low_freq = 100 / nyquist
        b_low, a_low = butter(2, low_freq, btype='low')
        audio_low = filtfilt(b_low, a_low, audio)

        # Combine: original + high boost - low cut
        audio_eq = audio + 0.3 * audio_high - 0.5 * audio_low

        return audio_eq

    def _dynamic_compression(self, audio: np.ndarray, threshold: float = 0.6, ratio: float = 4.0) -> np.ndarray:
        """Dynamic range compression for better speech intelligibility."""
        # Simple compressor
        compressed = np.copy(audio)

        # Find samples above threshold
        over_threshold = np.abs(audio) > threshold

        # Apply compression
        sign = np.sign(audio[over_threshold])
        magnitude = np.abs(audio[over_threshold])

        # Compress
        compressed_magnitude = threshold + (magnitude - threshold) / ratio
        compressed[over_threshold] = sign * compressed_magnitude

        return compressed

    def postprocess_text(self, text: str) -> str:
        """Postprocess text: spell correction."""
        if not self.enable_postprocessing:
            return text
        import re
        # Preserve speaker prefix like 'SPEAKER_01|' if present
        prefix = ''
        body = text
        if '|' in text:
            parts = text.split('|', 1)
            prefix = parts[0] + '|'
            body = parts[1].strip()

        # Load user-defined corrections from configs/post_corrections.yaml if exists
        corrections_path = os.path.join(CONFIGS_DIR, 'post_corrections.yaml')
        try:
            if os.path.exists(corrections_path):
                corrections = load_yaml(corrections_path)
            else:
                corrections = {}
        except Exception:
            corrections = {}

        # Apply regex replacements from config (keys are regex patterns)
        for pattern, repl in (corrections.items() if isinstance(corrections, dict) else []):
            try:
                body = re.sub(pattern, repl, body, flags=re.IGNORECASE)
            except re.error:
                # fallback to simple replace if pattern invalid
                body = body.replace(pattern, repl)

        # Collapse adjacent duplicate words (e.g., "я я" -> "я") while keeping punctuation
        body = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", body, flags=re.IGNORECASE)

        # Basic punctuation fixes: ensure commas after short phrases where a pause exists
        body = re.sub(r"\s+,", ",", body)
        body = re.sub(r",\s*([а-яА-Яa-zA-Z0-9])", r", \1", body)

        # Finally, try spell correction for small typos
        try:
            from spellchecker import SpellChecker
            spell = SpellChecker(language='ru')  # Assuming Russian text
            corrected_tokens = []
            for token in body.split():
                # Skip tokens that contain non-alphabetic chars (timestamps, punctuation)
                if re.search(r"[^\w\-']", token):
                    corrected_tokens.append(token)
                    continue
                corrected = spell.correction(token)
                corrected_tokens.append(corrected if corrected else token)
            body = ' '.join(corrected_tokens)
        except Exception:
            # Spellchecker absent or failed: ignore
            pass

        return prefix + body

    def transcribe_with_silero(self, audio: np.ndarray, language: str = 'ru') -> str:
        """Alternative transcription using Silero STT."""
        try:
            from silero import silero_stt

            model, decoder, utils = silero_stt(language=language)
            (read_batch, split_into_batches, read_audio, prepare_model_input) = utils

            input_audio = prepare_model_input([torch.from_numpy(audio)], device=torch.device('cpu'))
            output = model(input_audio)
            text = decoder(output[0])
            return text if text else ""
        except ImportError:
            logger.warning("Silero not installed. Skipping Silero STT.")
            return ""

        self.model = None
        self.current_model_size = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.device = self.get_device()
        self.available_compute_types = self.get_available_compute_type()
        self.current_compute_type = self.get_compute_type()

    @abstractmethod
    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   params: Optional[WhisperParams] = None,
                   ) -> Tuple[List[Segment], float]:
        """Inference whisper model to transcribe"""
        pass

    @abstractmethod
    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """Initialize whisper model"""
        pass

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            progress: gr.Progress = gr.Progress(),
            file_format: str = "SRT",
            add_timestamp: bool = True,
            progress_callback: Optional[Callable] = None,
            *pipeline_params,
            ) -> Tuple[List[Segment], float]:
        """
        Run transcription with conditional pre-processing and post-processing.
        The VAD will be performed to remove noise from the audio input in pre-processing, if enabled.
        The diarization will be performed in post-processing, if enabled.
        Due to the integration with gradio, the parameters have to be specified with a `*` wildcard.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        file_format: str
            Subtitle file format between ["SRT", "WebVTT", "txt", "lrc"]
        add_timestamp: bool
            Whether to add a timestamp at the end of the filename.
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.

        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class.
            This must be provided as a List with * wildcard because of the integration with gradio.
            See more info at : https://github.com/gradio-app/gradio/issues/2471

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        start_time = time.time()

        if not validate_audio(audio):
            return [Segment()], 0.0

        params = TranscriptionPipelineParams.from_list(list(pipeline_params))
        params = self.validate_gradio_values(params)
        bgm_params, vad_params, whisper_params, diarization_params = params.bgm_separation, params.vad, params.whisper, params.diarization

        # Preprocess audio if enabled
        if isinstance(audio, np.ndarray) and self.enable_preprocessing:
            progress(0, desc="Preprocessing audio...")
            audio = self.preprocess_audio(audio)

        if bgm_params.is_separate_bgm:
            music, audio, _ = self.music_separator.separate(
                audio=audio,
                model_name=bgm_params.uvr_model_size,
                device=bgm_params.uvr_device,
                segment_size=bgm_params.segment_size,
                save_file=bgm_params.save_file,
                progress=progress
            )

            if audio.ndim >= 2:
                audio = audio.mean(axis=1)
                if self.music_separator.audio_info is None:
                    origin_sample_rate = 16000
                else:
                    origin_sample_rate = self.music_separator.audio_info.sample_rate
                audio = self.resample_audio(audio=audio, original_sample_rate=origin_sample_rate)

            if bgm_params.enable_offload:
                self.music_separator.offload()
            elapsed_time_bgm_sep = time.time() - start_time

        origin_audio = deepcopy(audio)

        if vad_params.vad_filter:
            progress(0, desc="Filtering silent parts from audio..")
            vad_options = VadOptions(
                threshold=vad_params.threshold,
                min_speech_duration_ms=vad_params.min_speech_duration_ms,
                max_speech_duration_s=vad_params.max_speech_duration_s,
                min_silence_duration_ms=vad_params.min_silence_duration_ms,
                speech_pad_ms=vad_params.speech_pad_ms
            )

            vad_processed, speech_chunks = self.vad.run(
                audio=audio,
                vad_parameters=vad_options,
                progress=progress
            )

            if vad_processed.size > 0:
                audio = vad_processed
            else:
                vad_params.vad_filter = False

        result, elapsed_time_transcription = self.transcribe(
            audio,
            progress,
            progress_callback,
            *whisper_params.to_list()
        )
        if whisper_params.enable_offload:
            self.offload()

        if vad_params.vad_filter:
            restored_result = self.vad.restore_speech_timestamps(
                segments=result,
                speech_chunks=speech_chunks if 'speech_chunks' in locals() else [],
            )
            if restored_result:
                result = restored_result
            else:
                logger.info("VAD detected no speech segments in the audio.")

        if diarization_params.is_diarize:
            progress(0.99, desc="Diarizing speakers..")
            # Load environment variables and get HF token
            from modules.utils.env_loader import load_env_file, get_hf_token
            load_env_file()
            
            # Use token from UI, environment, or .env file
            token = diarization_params.hf_token or get_hf_token()
            
            # For songs and music content, use original audio for better diarization quality
            # The alignment is handled internally by the assign_word_speakers function
            diarizer_audio = origin_audio
            print(f"[diarization] Using original audio for better diarization quality")
            result, elapsed_time_diarization = self.diarizer.run(
                audio=diarizer_audio,
                use_auth_token=token if token else "",
                transcribed_result=result,
                device=diarization_params.diarization_device
            )
            if diarization_params.enable_offload:
                self.diarizer.offload()

        # Postprocess text if enabled
        if self.enable_postprocessing:
            for segment in result:
                if segment.text:
                    segment.text = self.postprocess_text(segment.text)

        self.cache_parameters(
            params=params,
            file_format=file_format,
            add_timestamp=add_timestamp
        )

        if not result:
            logger.info(f"Whisper did not detected any speech segments in the audio.")
            result = [Segment()]

        progress(1.0, desc="Finished.")
        total_elapsed_time = time.time() - start_time
        return result, total_elapsed_time

    def transcribe_file(self,
                        files: Optional[List] = None,
                        input_folder_path: Optional[str] = None,
                        include_subdirectory: Optional[str] = None,
                        save_same_dir: Optional[str] = None,
                        file_format: str = "SRT",
                        add_timestamp: bool = True,
                        progress=gr.Progress(),
                        *pipeline_params,
                        ) -> Tuple[str, List]:
        """
        Write subtitle file from Files

        Parameters
        ----------
        files: list
            List of files to transcribe from gr.Files()
        input_folder_path: Optional[str]
            Input folder path to transcribe from gr.Textbox(). If this is provided, `files` will be ignored and
            this will be used instead.
        include_subdirectory: Optional[str]
            When using `input_folder_path`, whether to include all files in the subdirectory or not
        save_same_dir: Optional[str]
            When using `input_folder_path`, whether to save output in the same directory as inputs or not, in addition
            to the original output directory. This feature is only available when using `input_folder_path`, because
            gradio only allows to use cached file path in the function yet.
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the subtitle filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                # Do not highlight words by default when generating subtitle files (avoids per-word <u> entries)
                # If downstream UI explicitly wants per-word highlights, this can be enabled there.
                "highlight_words": False
            }

            if input_folder_path:
                files = get_media_files(input_folder_path, include_sub_directory=bool(include_subdirectory))
            if isinstance(files, str):
                files = [files]
            if files and not isinstance(files[0], str):
                files = [file.name for file in files if hasattr(file, 'name')]

            files_info = {}
            for file in files:
                transcribed_segments, time_for_task = self.run(
                    file,
                    progress,
                    file_format,
                    add_timestamp,
                    None,
                    *pipeline_params,
                )

                file_name, file_ext = os.path.splitext(os.path.basename(file))
                if save_same_dir and input_folder_path:
                    output_dir = os.path.dirname(file)
                    subtitle, file_path = generate_file(
                        output_dir=output_dir,
                        output_file_name=file_name,
                        output_format=file_format,
                        result=transcribed_segments,
                        add_timestamp=add_timestamp,
                        **writer_options
                    )

                subtitle, file_path = generate_file(
                    output_dir=self.output_dir,
                    output_file_name=file_name,
                    output_format=file_format,
                    result=transcribed_segments,
                    add_timestamp=add_timestamp,
                    **writer_options
                )
                files_info[file_name] = {"subtitle": read_file(file_path), "time_for_task": time_for_task, "path": file_path}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{info["subtitle"]}'
                total_time += info["time_for_task"]

            result_str = f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
            result_file_path = [info['path'] for info in files_info.values()]

            return result_str, result_file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing file: {e}") from e

    def transcribe_mic(self,
                       mic_audio: str,
                       file_format: str = "SRT",
                       add_timestamp: bool = True,
                       progress=gr.Progress(),
                       *pipeline_params,
                       ) -> Tuple[str, str]:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        mic_audio: str
            Audio file path from gr.Microphone()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio..")
            transcribed_segments, time_for_task = self.run(
                mic_audio,
                progress,
                file_format,
                add_timestamp,
                None,
                *pipeline_params,
            )
            progress(1, desc="Completed!")

            file_name = "Mic"
            subtitle, file_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_format,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
                **writer_options
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return result_str, file_path
        except Exception as e:
            raise RuntimeError(f"Error transcribing mic: {e}") from e

    def transcribe_youtube(self,
                           youtube_link: str,
                           file_format: str = "SRT",
                           add_timestamp: bool = True,
                           progress=gr.Progress(),
                           *pipeline_params,
                           ) -> Tuple[str, str]:
        """
        Write subtitle file from Youtube

        Parameters
        ----------
        youtube_link: str
            URL of the Youtube video to transcribe from gr.Textbox()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtube_link)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.run(
                audio,
                progress,
                file_format,
                add_timestamp,
                None,
                *pipeline_params,
            )

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle, file_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_format,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
                **writer_options
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"

            if os.path.exists(audio):
                os.remove(audio)

            return result_str, file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing youtube: {e}") from e

    def get_compute_type(self):
        if "float16" in self.available_compute_types:
            return "float16"
        if "float32" in self.available_compute_types:
            return "float32"
        else:
            return self.available_compute_types[0]

    def get_available_compute_type(self):
        if self.device == "cuda":
            return list(ctranslate2.get_supported_compute_types("cuda"))
        else:
            return list(ctranslate2.get_supported_compute_types("cpu"))

    def offload(self):
        """Offload the model and free up the memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        gc.collect()

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        """
        Get {hours} {minutes} {seconds} time format string

        Parameters
        ----------
        elapsed_time: str
            Elapsed time for transcription

        Returns
        ----------
        Time format string
        """
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours:
            time_str += f"{hours} hours "
        if minutes:
            time_str += f"{minutes} minutes "
        seconds = round(seconds)
        time_str += f"{seconds} seconds"

        return time_str.strip()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            if not BaseTranscriptionPipeline.is_sparse_api_supported():
                # Device `SparseMPS` is not supported for now. See : https://github.com/pytorch/pytorch/issues/87886
                return "cpu"
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def is_sparse_api_supported():
        if not torch.backends.mps.is_available():
            return False

        try:
            device = torch.device("mps")
            sparse_tensor = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1], [2, 3]]),
                values=torch.tensor([1, 2]),
                size=(4, 4),
                device=device
            )
            return True
        except RuntimeError:
            return False

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        """Remove gradio cached files"""
        if not file_paths:
            return

        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.error(f"Error removing file {file_path}: {e}")

    @staticmethod
    def validate_gradio_values(params: TranscriptionPipelineParams):
        """
        Validate gradio specific values that can't be displayed as None in the UI.
        Related issue : https://github.com/gradio-app/gradio/issues/8723
        """
        if params.whisper.lang is None:
            pass
        elif params.whisper.lang == AUTOMATIC_DETECTION:
            params.whisper.lang = None
        else:
            language_code_dict = {value: key for key, value in tokenizer.LANGUAGES.items()}
            if params.whisper.lang in language_code_dict:
                params.whisper.lang = language_code_dict[params.whisper.lang]

        if params.whisper.initial_prompt == GRADIO_NONE_STR:
            params.whisper.initial_prompt = None
        if params.whisper.prefix == GRADIO_NONE_STR:
            params.whisper.prefix = None
        if params.whisper.hotwords == GRADIO_NONE_STR:
            params.whisper.hotwords = None
        if params.whisper.max_new_tokens == GRADIO_NONE_NUMBER_MIN:
            params.whisper.max_new_tokens = None
        if params.whisper.hallucination_silence_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.hallucination_silence_threshold = None
        if params.whisper.language_detection_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.language_detection_threshold = None
        if params.vad.max_speech_duration_s == GRADIO_NONE_NUMBER_MAX:
            params.vad.max_speech_duration_s = float('inf')
        return params

    @staticmethod
    def cache_parameters(
        params: TranscriptionPipelineParams,
        file_format: str = "SRT",
        add_timestamp: bool = True
    ):
        """Cache parameters to the yaml file"""
        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        param_to_cache = params.to_dict()

        cached_yaml = {**cached_params, **param_to_cache}
        cached_yaml["whisper"]["add_timestamp"] = add_timestamp
        cached_yaml["whisper"]["file_format"] = file_format

        supress_token = cached_yaml["whisper"].get("suppress_tokens", None)
        if supress_token and isinstance(supress_token, list):
            cached_yaml["whisper"]["suppress_tokens"] = str(supress_token)

        if cached_yaml["whisper"].get("lang", None) is None:
            cached_yaml["whisper"]["lang"] = AUTOMATIC_DETECTION.unwrap()
        else:
            language_dict = tokenizer.LANGUAGES
            cached_yaml["whisper"]["lang"] = language_dict[cached_yaml["whisper"]["lang"]]

        if cached_yaml["vad"].get("max_speech_duration_s", float('inf')) == float('inf'):
            cached_yaml["vad"]["max_speech_duration_s"] = GRADIO_NONE_NUMBER_MAX

        if cached_yaml is not None and cached_yaml:
            save_yaml(cached_yaml, DEFAULT_PARAMETERS_CONFIG_PATH)

    @staticmethod
    def resample_audio(audio: Union[str, np.ndarray],
                       new_sample_rate: int = 16000,
                       original_sample_rate: Optional[int] = None,) -> np.ndarray:
        """Resamples audio to 16k sample rate, standard on Whisper model"""
        if isinstance(audio, str):
            audio, original_sample_rate = torchaudio.load(audio)
        else:
            if original_sample_rate is None:
                raise ValueError("original_sample_rate must be provided when audio is numpy array.")
            audio = torch.from_numpy(audio)
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
        resampled_audio = resampler(audio).numpy()
        return resampled_audio
