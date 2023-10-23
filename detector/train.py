import torch as tr
import lightning.pytorch as trl, torchmetrics as trm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .loss import AsymmetricLoss

class TrainSupervised(trl.LightningModule):

    def __init__(
        self,
        data,
        model,
        thresh=0.5,
        learning_rate=1e-3,
        weight_decay=1e-2,
        momentum=0.9,
        augm_cutmix=False,
        asl_config={
            'gamma_neg': 1,
            'gamma_pos': 0,
        },
    ):
        super().__init__()
        assert model.n_class == data.dataset.n_class
        n_class = model.n_class
        self.data = data
        self.model = model
        self.criterion = AsymmetricLoss(**asl_config)
        self.metrics = trm.MetricCollection([
            trm.classification.MultilabelPrecision(n_class, thresh, 'macro'),
            trm.classification.MultilabelRecall(n_class, thresh, 'macro'),
            trm.classification.MultilabelF1Score(n_class, thresh, 'macro'),
            trm.classification.MultilabelAveragePrecision(n_class, 'macro', 10),
            trm.classification.MultilabelAUROC(n_class, 'macro', 10),
        ])
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')
        self.save_hyperparameters(ignore=('data', 'model'))

    def forward(self, x):
        return self.model(x)

    def evaluate(self, batch, stage, metrics):
        x, y = batch
        yh = self(x)
        loss = self.criterion(yh, y.type(tr.float))
        metr = metrics(yh, y.type(tr.int))

        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(metr, on_step=False, on_epoch=True, prog_bar=False)

    def training_step(self, batch, _idx):
        x, y = batch
        yh = self(x)

        if self.hparams.augm_cutmix:
            l1, l2, lam = y
            loss = lam * self.criterion(yh, l1) + (1 - lam) * self.criterion(yh, l2)
        else:
            loss = self.criterion(yh, y.type(tr.float))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _idx):
        self.evaluate(batch, 'val', self.val_metrics)

    def test_step(self, batch, _idx):
        self.evaluate(batch, 'test', self.test_metrics)

    def configure_optimizers(self):
        optim = AdamW(
            self.parameters(),
            amsgrad=True,
            betas=(0.9, 0.999),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        schd = {
            'scheduler': OneCycleLR(
                optim,
                self.hparams.learning_rate,
                anneal_strategy='cos',
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.data.train_dataloader()),
            ),
            'interval': 'step',
        }

        return {'optimizer': optim, 'lr_scheduler': schd}
