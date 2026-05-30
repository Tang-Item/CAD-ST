import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*GradScaler.*is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*does not have many workers.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Experiment logs directory.*exists and is not empty.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Checkpoint directory .* exists and is not empty.*", category=UserWarning)

from herst import ViT_HER2ST, ViT_SKIN
from model import CAD_ST
from pytorch_lightning.loggers import CSVLogger
import torch
import os
import shutil
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import parser_option, apply_dataset_defaults, seed_torch, effective_train_seed

def _num_folds_for_dataset(dataset: str) -> int:
    if dataset == "her2st":
        return 32

    return 12

def train(args):
    torch.set_float32_matmul_precision('high')
    i = int(args.fold)
    base_seed = getattr(args, "seed", None)
    if base_seed is not None:
        seed_torch(effective_train_seed(base_seed, i))
    args.divide_size = 1
    output_dir = getattr(args, 'output_dir', 'output')
    fold_dir = os.path.join(output_dir, args.dataset, f"fold{i}")
    cache_dir = getattr(args, 'cache_dir', None) or os.path.join(fold_dir, "cache")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fold_dir, exist_ok=True)

    shutil.rmtree(os.path.join(fold_dir, "logs"), ignore_errors=True)
    for fn in os.listdir(fold_dir):
        if fn.endswith(".ckpt"):
            try:
                os.remove(os.path.join(fold_dir, fn))
            except OSError:
                pass

    val_checkpoint_callback = ModelCheckpoint(
        dirpath=fold_dir,
        filename=f"fold{i}_best_model",
        monitor="pcc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="pcc",
        mode="max",
        patience=50,
    )

    csv_logger = CSVLogger(save_dir=fold_dir, name="logs", version=0)

    if args.dataset == "her2st":
        pm = getattr(args, "pos_mode", "raw")
        train_data = ViT_HER2ST(train=True, flatten=False, ori=True, adj=False, fold=i,
                                use_cache_matrices=getattr(args, 'use_cache_matrices', False),
                                cache_dir=cache_dir, pos_mode=pm)
        val_data = ViT_HER2ST(train=False, flatten=False, ori=True, adj=False, fold=i,
                              use_cache_matrices=getattr(args, 'use_cache_matrices', False),
                              cache_dir=cache_dir, pos_mode=pm)
        args.dim_out = 785
    else:
        args.dim_out = 171
        train_data = ViT_SKIN(train=True, flatten=False, ori=True, adj=False, fold=i,
                              use_cache_matrices=getattr(args, 'use_cache_matrices', False),
                              cache_dir=cache_dir)
        val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=i,
                            use_cache_matrices=getattr(args, 'use_cache_matrices', False),
                            cache_dir=cache_dir)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

    args.iter_per_epoch = len(train_loader)

    model = CAD_ST(args=args)

    trainer = pl.Trainer(logger=csv_logger, precision=32, max_epochs=args.epochs,
                         accelerator='gpu', devices=[args.device_id],
                         callbacks=[val_checkpoint_callback, early_stopping_callback],
                         gradient_clip_val=1.0,
                         log_every_n_steps=5)

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    args = parser_option()
    datasets = [args.dataset]
    if getattr(args, "all_datasets", False):
        datasets = ["her2st", "cSCC"]

    for ds in datasets:
        args.dataset = ds
        apply_dataset_defaults(args)
        if getattr(args, "all_folds", False):
            n_folds = int(getattr(args, "num_folds", 0) or _num_folds_for_dataset(args.dataset))
            for f in range(n_folds):
                args.fold = f
                train(args)
        else:
            train(args)
