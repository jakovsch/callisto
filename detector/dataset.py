import gzip, numpy as np, torch as tr
import lightning.pytorch as trl
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
from torch.distributions.uniform import Uniform
from torchaudio.transforms import SpecAugment
from torchvision.transforms import v2 as tvt
from astropy.io import fits

from .utils.gather import DATADIR, load_fits, load_lists, select_dataset

class CallistoDataModule(trl.LightningDataModule):

    def __init__(
        self,
        n_worker=6,
        batch_size=24,
        dset_config={},
        img_size=(300, 600),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = CallistoDataset(**dset_config)

    def prepare_data(self):
        return

    def setup(self, stage=None):
        train, val, test = self.dataset.split()
        transforms = [
            tvt.ToImage(),
            tvt.ToDtype(tr.float32, scale=True),
            tvt.Resize(
                self.hparams.img_size,
                interpolation=tvt.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            DecibelScale(),
            SpecNorm(),
        ]
        train_transforms = [
            RandomRoll(0.0, 0.9, p=0.6),
            SpecAugment(4, 30, 3, 30, zero_masking=False, p=1.0),
        ]
        self.train = TransformDataset(train, tvt.Compose(transforms+train_transforms))
        self.val = TransformDataset(val, tvt.Compose(transforms))
        self.test = TransformDataset(test, tvt.Compose(transforms))

    def train_dataloader(self):
        mask = self.dataset.flist.index.isin(self.train.dataset.indices)
        weights = self.dataset.event_weights.loc[mask].values
        return DataLoader(
            self.train,
            shuffle=False,
            persistent_workers=True,
            sampler=WeightedRandomSampler(
                weights, len(self.train), replacement=True,
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            shuffle=False,
            persistent_workers=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            shuffle=False,
            persistent_workers=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.n_worker,
        )

class CallistoDataset(Dataset):

    def __init__(
        self,
        path=DATADIR,
        **kwargs,
    ):
        self.blist = select_dataset(load_lists(path), **kwargs)
        self.flist = load_fits(self.blist, path)
        self.classes = self.blist['type'].array.categories
        eblist = self.blist.explode('stat')
        self.n_burst = len(self.blist)
        self.n_event = len(eblist)
        self.n_files = len(self.flist)
        self.n_class = len(self.classes)
        self.class_weights = 1.0 / eblist['type'].value_counts()
        self.event_weights = self.flist['evnt'].map(self.multilabel_weight)
        self.labels = self.flist['evnt'].map(self.types_to_label)

    def __getitem__(self, idx):
        event = self.flist.iloc[idx]
        target = self.labels.iloc[idx]
        with gzip.open(event.path) as f:
            with fits.open(f, uint=True) as (img, axes):
                return img.data[..., np.newaxis], target

    def __len__(self):
        return self.n_files

    def types_to_label(self, eidx):
        types = self.blist.loc[eidx]['type'].array.codes
        label = np.zeros(self.n_class, dtype=np.float32)
        label.put(types, 1.0)
        return label

    def multilabel_weight(self, eidx):
        # TODO balanced per-class, pos/neg
        types = self.blist.loc[eidx]['type'].array
        return np.mean(self.class_weights.loc[types].values)

    def split(self, frac=[0.8, 0.5]):
        # TODO stratified sampling, rng seed
        rem = self.flist
        for f in frac:
            spl = rem.sample(frac=f, replace=False)
            idxs = spl.index
            rem = rem.loc[~rem.index.isin(idxs)]
            idxr = rem.index
            yield Subset(self, idxs)
        yield Subset(self, idxr)

class TransformDataset(Dataset):

    def __init__(
        self,
        dataset,
        transform=None,
    ):
        self.dataset = dataset
        self.transform = transform or (lambda x: x)

    #def __getattr__(self, attr):
    #    return getattr(self.dataset, attr)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return self.transform(data), target

    def __len__(self):
        return len(self.dataset)

class DecibelScale(tvt.Transform):

    def forward(self, data):
        return (data - tr.min(data)) * 2500.0 / 25.4

class SpecNorm(tvt.Transform):

    def __init__(
        self,
        min=-1.0,
        max=4.0,
    ):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, sgram):
        filt = sgram - tr.median(sgram, -1, keepdims=True).values
        clip = tr.clamp(filt, self.min, self.max)
        norm = (clip - self.min) / (self.max - self.min)
        return (norm - tr.mean(norm)) / tr.std(norm)

class RandomRoll(tvt.Transform):

    def __init__(
        self,
        min=0.0,
        max=1.0,
        p=0.5,
    ):
        super().__init__()
        self.dist = Uniform(min, max)
        self.p = p

    def forward(self, sgram):
        w = sgram.shape[-1]
        shift = int(w * self.dist.sample() * (tr.rand(1) < self.p))
        return tr.roll(sgram, shift, -1)
