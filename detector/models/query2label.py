import torch as tr, torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer

class Query2Label(nn.Module):
    """https://arxiv.org/abs/2107.10834"""

    def __init__(
        self,
        extractor,
        n_class,
        n_feat,
        n_hidden=256,
        heads=8,
        encoders=6,
        decoders=6,
        position_enc=True,
    ):
        super().__init__()
        self.extractor = extractor
        self.n_class = n_class
        self.n_feat = n_feat
        self.n_hidden = n_hidden
        self.position_enc = position_enc

        self.conv = nn.Conv2d(n_feat, n_hidden, 1)
        self.xfrm = nn.Transformer(n_hidden, heads, encoders, decoders)

        if position_enc:
            self.pos_enc = PositionalEncodingPermute2D(n_hidden)
            self.pos_enc_add = Summer(self.pos_enc)

        self.clsf = nn.Linear(n_class*n_hidden, n_class)
        self.lemb = nn.Parameter(tr.rand(1, n_class, n_hidden))

    def forward(self, x):
        out = self.extractor.forward_features(x)
        feat = self.conv(out)
        bsize = feat.shape[0]

        if self.position_enc:
            feat = self.pos_enc_add(feat*0.1)

        feat = feat.flatten(2).permute(2, 0, 1)
        lemb = self.lemb.repeat(bsize, 1, 1)
        lemb = lemb.transpose(0, 1)
        feat = self.xfrm(feat, lemb).transpose(0, 1)
        feat = tr.reshape(feat, (bsize, self.n_class*self.n_hidden))

        return self.clsf(feat)
