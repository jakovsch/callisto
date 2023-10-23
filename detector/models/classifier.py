import torch.nn as nn, timm
from .query2label import Query2Label

class MultilabelClassifier(nn.Module):

    def __init__(
        self,
        model,
        n_class,
        ext_config={
            #'img_size'
            #'window_size'
            #'embed_dim'
            'in_chans': 1,
            'drop_rate': 0,
            'pretrained': True,
            'exportable': True,
        },
        q2l_config={},
    ):
        super().__init__()
        extractor = timm.create_model(model, **ext_config)
        n_feat = extractor.get_classifier().in_features
        self.n_class = n_class
        self.model = Query2Label(extractor, n_class, n_feat, **q2l_config)

    def forward(self, x):
        return self.model(x)
