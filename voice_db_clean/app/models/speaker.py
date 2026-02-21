from speechbrain.inference import EncoderClassifier
import torch

class SpeakerEncoder:
    def __init__(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa_voxceleb",
            run_opts={"device": "cpu"}
        )

    def encode(self, waveform):
        with torch.no_grad():
            emb = self.model.encode_batch(torch.tensor(waveform))
        return emb.squeeze().numpy()
