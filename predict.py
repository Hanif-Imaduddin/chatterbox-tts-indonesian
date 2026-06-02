import torch
import torchaudio as ta
import tempfile
from cog import BasePredictor, Input, Path
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL_REPO          = "grandhigh/Chatterbox-TTS-Indonesian"
CHECKPOINT_FILENAME = "t3_cfg.safetensors"


class Predictor(BasePredictor):
    def setup(self):
        """Load model — dijalankan sekali saat container startup."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[setup] device: {device}")

        self.model = ChatterboxTTS.from_pretrained(device=device)

        ckpt = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILENAME)
        self.model.t3.load_state_dict(load_file(ckpt, device="cpu"))

        self.device = device
        torch.cuda.empty_cache()
        print("[setup] Model siap!")

    def predict(
        self,
        text: str = Input(
            description="Teks Bahasa Indonesia yang akan diucapkan.",
            default="Halo, selamat datang! Ini adalah demo Text-to-Speech Bahasa Indonesia."
        ),
        audio_prompt: Path = Input(
            description="(Opsional) Audio referensi untuk voice cloning (.wav/.mp3, 5-15 detik).",
            default=None
        ),
        exaggeration: float = Input(
            description="Ekspresi emosi: 0.0 = monoton, 1.0 = sangat ekspresif.",
            default=0.5, ge=0.0, le=1.0
        ),
        cfg_weight: float = Input(
            description="CFG weight. Turunkan ke ~0.3 jika referensi bicara cepat.",
            default=0.5, ge=0.0, le=1.0
        ),
    ) -> Path:
        torch.cuda.empty_cache()
        kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
        if audio_prompt is not None:
            kwargs["audio_prompt_path"] = str(audio_prompt)

        wav = self.model.generate(text, **kwargs)
        out = Path(tempfile.mktemp(suffix=".wav"))
        ta.save(str(out), wav, self.model.sr)
        return out
