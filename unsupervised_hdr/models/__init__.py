import timm
from torch import nn

from .decoder import SimpleDecoder
from .encoder import SimpleEncoder
from .loss import ImageSpaceLoss

encoder_factory = {"SimpleEncoder": SimpleEncoder}
decoder_factory = {"SimpleDecoder": SimpleDecoder}


class EncoderDecoderModel(nn.Module):
    def __init__(
        self, encoder: str, decoder: str, encoder_pretrained: bool = False
    ) -> None:
        super().__init__()
        if encoder == "SimpleEncoder":
            self.encoder = SimpleEncoder()
        else:
            self.encoder = timm.create_model(
                encoder,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                pretrained=encoder_pretrained,
            )
        if decoder == "SimpleDecoder":
            self.decoder = SimpleDecoder(feature_info=self.encoder.feature_info)

    def forward(self, x):
        return self.decoder(self.encoder(x))
