import torch
import torch.nn as nn
from hifigan.generator import HiFTGenerator
from hifigan.f0_predictor import ConvRNNF0Predictor


class Cosy24kVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hifigan_generator = HiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=24000,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            f0_predictor=ConvRNNF0Predictor(
                num_class=1,
                in_channels=80,
                cond_channels=512,
            ),
        )
    
    def decode(self, mel, device="cuda"):
        """
        Args: mel: (batch_size, n_frames, n_mel)
        """
        generated_speech, f0 = self.hifigan_generator.forward(
                {"speech_feat": mel.transpose(1, 2)}, device=device
            )
        return generated_speech

    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load a pretrained model from a checkpoint."""
        model = cls()
        model.hifigan_generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
