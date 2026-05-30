import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*GradScaler.*is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*does not have many workers.*", category=UserWarning)

from herst import ViT_HER2ST, ViT_SKIN
from model import CAD_ST
import torch
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import parser_option, apply_dataset_defaults

def predict(args):
    torch.set_float32_matmul_precision('high')
    output_dir = getattr(args, 'output_dir', 'output')
    fold_dir = os.path.join(output_dir, args.dataset, f"fold{args.fold}")
    cache_dir = getattr(args, 'cache_dir', None) or os.path.join(fold_dir, "cache")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fold_dir, exist_ok=True)
    if args.dataset == "her2st":
        val_data = ViT_HER2ST(train=False, flatten=False, ori=True, adj=False, fold=args.fold,
                              use_cache_matrices=getattr(args, 'use_cache_matrices', False),
                              cache_dir=cache_dir,
                              pos_mode=getattr(args, 'pos_mode', 'raw'))
        args.dim_out = 785
    else:
        val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=args.fold,
                            use_cache_matrices=getattr(args, 'use_cache_matrices', False),
                            cache_dir=cache_dir)
        args.dim_out = 171

    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

    ckpt_candidates = [
        os.path.join(fold_dir, f"fold{args.fold}_best_model.ckpt"),
        os.path.join(fold_dir, "last.ckpt"),
    ]
    ckpt_dir = next((p for p in ckpt_candidates if os.path.exists(p)), ckpt_candidates[0])
    print(f"Loading checkpoints from {ckpt_dir}")

    model = CAD_ST(args)

    trainer = pl.Trainer(precision=32, max_epochs=args.epochs,
                         accelerator='gpu', devices=[args.device_id])
    trainer.test(model, val_loader, ckpt_path=ckpt_dir)

if __name__ == "__main__":
    args = parser_option()
    apply_dataset_defaults(args)
    predict(args)
