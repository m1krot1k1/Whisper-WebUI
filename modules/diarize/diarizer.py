import os
import torch
from typing import List, Union, BinaryIO, Optional, Tuple
import numpy as np
import time
import logging
import gc

from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.diarize_pipeline import DiarizationPipeline, assign_word_speakers
from modules.diarize.audio_loader import load_audio
from modules.whisper.data_classes import *
from modules.utils.env_loader import load_env_file, get_hf_token
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backend'))
from common.config_loader import load_server_config


class Diarizer:
    def __init__(self,
                 model_dir: str = DIARIZATION_MODELS_DIR
                 ):
        self.device = self.get_device()
        self.available_device = self.get_available_device()
        self.compute_type = "float16"
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.pipe = None
        # Load configuration
        self.config = load_server_config()
        self.diarization_config = self.config.get('diarization', {})
        
    def _get_merge_settings(self, raw_segments: List[dict]) -> dict:
        """Get merge settings based on content type (music vs speech)"""
        merge_settings = self.diarization_config.get('merge_settings', {})
        music_keywords = self.diarization_config.get('music_keywords', [])
        
        # Check if content appears to be music
        is_music_content = any(
            word.lower() in (seg.get("text", "") or "").lower() 
            for seg in raw_segments[:5]  # Check first 5 segments
            for word in music_keywords
        ) if raw_segments else False
        
        if is_music_content:
            settings = {
                'merge_gap': merge_settings.get('music_merge_gap', 0.2),
                'min_segment_length': merge_settings.get('music_min_segment_length', 0.3),
                'max_segment_length': merge_settings.get('music_max_segment_length', 8.0)
            }
            print("[diarization] Detected music content, using conservative merge settings")
        else:
            settings = {
                'merge_gap': merge_settings.get('default_merge_gap', 0.3),
                'min_segment_length': merge_settings.get('min_segment_length', 0.5),
                'max_segment_length': merge_settings.get('max_segment_length', 15.0)
            }
        
        print(f"[diarization] Using merge_gap={settings['merge_gap']}s, max_length={settings['max_segment_length']}s")
        return settings

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            transcribed_result: List[Segment],
            use_auth_token: str,
            device: Optional[str] = None
            ) -> Tuple[List[Segment], float]:
        """
        Diarize transcribed result as a post-processing

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        transcribed_result: List[Segment]
            transcribed result through whisper.
        use_auth_token: str
            Huggingface token with READ permission. This is only needed the first time you download the model.
            You must manually go to the website https://huggingface.co/pyannote/speaker-diarization-3.1 and agree to their TOS to download the model.
        device: Optional[str]
            Device for diarization.

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        start_time = time.time()

        if device is None:
            device = self.device

        if device != self.device or self.pipe is None:
            self.update_pipe(
                device=device,
                use_auth_token=use_auth_token
            )

        audio = load_audio(audio)

        diarization_segments = self.pipe(audio)
        diarized_result = assign_word_speakers(
            diarization_segments,
            {"segments": transcribed_result}
        )

        # Merge adjacent segments with same speaker to avoid fragmentation into one-word lines
        raw_segments = diarized_result.get("segments", [])
        merged = []
        
        # Get merge settings based on content type
        settings = self._get_merge_settings(raw_segments)
        merge_gap = settings['merge_gap']
        min_segment_length = settings['min_segment_length']
        max_segment_length = settings['max_segment_length']

        for seg in raw_segments:
            spk = seg.get("speaker", "None")
            text = (seg.get("text") or "").strip()
            start = seg.get("start")
            end = seg.get("end")

            if not merged:
                merged.append({"speaker": spk, "text": text, "start": start, "end": end})
                continue

            last = merged[-1]
            # Check if merging would create too long segment
            current_length = (end - start) if (start is not None and end is not None) else 0
            last_length = (last["end"] - last["start"]) if (last.get("start") is not None and last.get("end") is not None) else 0
            would_be_too_long = (last_length + current_length) > max_segment_length
            
            # if same speaker and small gap and not too long, merge texts
            if (last["speaker"] == spk and 
                (start is not None and last["end"] is not None and start - last["end"] <= merge_gap) and
                not would_be_too_long):
                # normalize whitespace and avoid adding exact duplicate tokens
                if text:
                    # if last text is empty, just setz nen yt gjybvf.? 
                    if not last["text"]:
                        last["text"] = text
                    else:
                        # avoid adding the same trailing word twice
                        if not last["text"].endswith(text):
                            last["text"] = (last["text"] + " " + text).strip()
                # extend end
                if end is not None:
                    last["end"] = end
            else:
                merged.append({"speaker": spk, "text": text, "start": start, "end": end})

        # Post-process: merge or attach very short segments to neighbors to avoid one-word fragments
        if len(merged) > 1:
            adjusted = []
            i = 0
            while i < len(merged):
                cur = merged[i]
                cur_len = None
                if cur.get("start") is not None and cur.get("end") is not None:
                    cur_len = cur["end"] - cur["start"]

                # if current segment is too short, try to merge with neighbor (prefer same speaker neighbors)
                if cur_len is not None and cur_len < min_segment_length:
                    merged_into_prev = False
                    # try previous if exists and same speaker
                    if adjusted:
                        prev = adjusted[-1]
                        if prev["speaker"] == cur["speaker"]:
                            # merge into previous
                            if cur.get("text"):
                                if not prev["text"]:
                                    prev["text"] = cur["text"]
                                elif not prev["text"].endswith(cur.get("text")):
                                    prev["text"] = (prev["text"] + " " + cur.get("text")).strip()
                            if cur.get("end") is not None:
                                prev["end"] = cur.get("end")
                            merged_into_prev = True
                    if not merged_into_prev and i + 1 < len(merged):
                        # merge into next
                        nxt = merged[i + 1]
                        if nxt["speaker"] == cur["speaker"]:
                            # attach to next
                            if cur.get("text"):
                                if not nxt["text"]:
                                    nxt["text"] = cur.get("text")
                                elif not nxt["text"].startswith(cur.get("text")):
                                    nxt["text"] = (cur.get("text") + " " + nxt.get("text")).strip()
                            if cur.get("start") is not None:
                                nxt["start"] = cur.get("start")
                            # skip adding cur, it'll be absorbed by next
                            i += 1
                            continue
                        else:
                            # attach short segment to previous even if speaker differs (as a fallback)
                            if adjusted:
                                prev = adjusted[-1]
                                if cur.get("text"):
                                    if not prev["text"]:
                                        prev["text"] = cur.get("text")
                                    elif not prev["text"].endswith(cur.get("text")):
                                        prev["text"] = (prev["text"] + " " + cur.get("text")).strip()
                                if cur.get("end") is not None:
                                    prev["end"] = cur.get("end")
                                merged_into_prev = True
                    if not merged_into_prev:
                        adjusted.append(cur)
                else:
                    adjusted.append(cur)
                i += 1
            merged = adjusted

        segments_result = []
        for segment in merged:
            speaker = segment.get("speaker", "None")
            diarized_text = speaker + "|" + (segment.get("text") or "").strip()
            segments_result.append(Segment(
                start=segment.get("start"),
                end=segment.get("end"),
                text=diarized_text
            ))

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_pipe(self,
                    use_auth_token: Optional[str] = None,
                    device: Optional[str] = None,
                    ):
        """
        Set pipeline for diarization

        Parameters
        ----------
        use_auth_token: str
            Huggingface token with READ permission. This is only needed the first time you download the model.
            You must manually go to the website https://huggingface.co/pyannote/speaker-diarization-3.1 and agree to their TOS to download the model.
        device: str
            Device for diarization.
        """
        if device is None:
            device = self.get_device()
        self.device = device

        os.makedirs(self.model_dir, exist_ok=True)

        # Load environment variables from .env file
        load_env_file()
        
        # Get token from parameter, environment, or .env file
        token = use_auth_token or get_hf_token()
        
        if (not os.listdir(self.model_dir) and not token):
            print(
                "\nFailed to diarize. You need huggingface token and agree to their requirements to download the diarization model.\n"
                "Go to \"https://huggingface.co/pyannote/speaker-diarization-3.1\" and follow their instructions to download the model.\n"
                "You can also set HF_TOKEN in your .env file or environment variables.\n"
            )
            return

        if token:
            print(f"Using Hugging Face token for diarization model download...")

        logger = logging.getLogger("speechbrain.utils.train_logger")
        # Disable redundant torchvision warning message
        logger.disabled = True
        self.pipe = DiarizationPipeline(
            use_auth_token=token,
            device=device,
            cache_dir=self.model_dir
        )
        logger.disabled = False

    def offload(self):
        """Offload the model and free up the memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        gc.collect()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def get_available_device():
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.xpu.is_available():
            devices.append("xpu")
        if torch.backends.mps.is_available():
            devices.append("mps")
        return devices