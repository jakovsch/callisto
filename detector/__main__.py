from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging

from .train import TrainSupervised
from .dataset import CallistoDataModule
from .models import MultilabelClassifier

seed_everything(42, True)

data = CallistoDataModule(
    n_worker=6,
    batch_size=24,
)
model = MultilabelClassifier(
    'focalnet_tiny_lrf.ms_in1k', 11,
)
task = TrainSupervised(
    data, model,
)
trainer = Trainer(
    max_epochs=20,
    log_every_n_steps=1,
    accumulate_grad_batches=1,
    precision='32-true',
    accelerator='auto',
    strategy='auto',
    devices='auto',
    callbacks=[
        ModelCheckpoint(
            every_n_epochs=1,
        ),
        StochasticWeightAveraging(
            0.04, swa_epoch_start=0.7,
        ),
    ],
)

trainer.fit(task, datamodule=data)
