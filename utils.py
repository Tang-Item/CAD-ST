import os
import sys
import argparse
import yaml
import types
import torch
import numpy as np
import random
from scipy.stats import pearsonr

def get_R(data1: torch.Tensor, data2: torch.Tensor, dim: int = 1, func=pearsonr, eps: float = 1e-12):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    r1, p1 = [], []

    n = int(data1.shape[dim])
    for g in range(n):
        if dim == 1:
            x = np.asarray(data1[:, g], dtype=np.float64)
            y = np.asarray(data2[:, g], dtype=np.float64)
        elif dim == 0:
            x = np.asarray(data1[g, :], dtype=np.float64)
            y = np.asarray(data2[g, :], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported dim={dim}")

        if (
            x.size < 2
            or y.size < 2
            or (not np.isfinite(x).all())
            or (not np.isfinite(y).all())
            or np.nanstd(x) < eps
            or np.nanstd(y) < eps
        ):
            r1.append(np.nan)
            p1.append(np.nan)
            continue

        r, pv = func(x, y)
        r1.append(r)
        p1.append(pv)
    return np.array(r1, dtype=np.float64), np.array(p1, dtype=np.float64)

def seed_torch(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

FOLD_SEED_STRIDE = 1_000_003

def effective_train_seed(base_seed: int, fold: int) -> int:
    return int(base_seed) + int(fold) * FOLD_SEED_STRIDE

DATASET_DEFAULTS = {
    "her2st": {
        "pos_mode": "raw",
        "attn_k": 16,
        "w_softcon": 0.02,
        "w_recon": 0.05,
        "w_zinb": 0.25,
        "gate_entropy_weight": 0,
        "tau": 0.15,
        "dist_sigma": 0.3,
        "dist_lambda": 0.2,
        "mask_rate": 0.1,
        "mask_block_size": 8,
        "ema_decay": 0.998,
        "lr": 1e-4,
    },

    "cSCC": {
        "pos_mode": "raw",
        "attn_k": 24,
        "w_softcon": 0.05,
        "w_recon": 0.08,
        "w_zinb": 0.55,
        "gate_entropy_weight": 0,
        "tau": 0.15,
        "dist_sigma": 0.3,
        "dist_lambda": 0.2,
        "mask_rate": 0.4,
        "mask_block_size": 8,
        "ema_decay": 0.998,
        "lr": 3e-4,
        "epochs": 250,
    },
}

def _arg_explicit(name: str) -> bool:
    flag = f"--{name}"
    return any(x == flag or x.startswith(f"{flag}=") for x in sys.argv)

def apply_dataset_defaults(args) -> None:
    ds = getattr(args, "dataset", "")
    d = DATASET_DEFAULTS.get(ds)
    if d is None:
        if getattr(args, "w_softcon", None) is None:
            args.w_softcon = 0.05
        if getattr(args, "w_recon", None) is None:
            args.w_recon = 0.05
    else:
        for key, value in d.items():
            if not _arg_explicit(key):
                setattr(args, key, value)

    args.soft_target_mode = "dist_only"

def dict_to_namespace(d):
    namespace = types.SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def parser_option():
    parser = argparse.ArgumentParser('gene prediction', add_help=False)
    parser.add_argument('--output_dir', type=str, default='output_cscc_w_recon_0p08_seed4343')
    parser.add_argument('--name', type=str, default='CAD-ST_model')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cSCC',
        choices=['her2st', 'cSCC'],
        help='her2st: ViT_HER2ST; cSCC: ViT_SKIN（12 folds, dim_out=171）',
    )
    parser.add_argument('--all_datasets',default=False,action=argparse.BooleanOptionalAction,
        help='Train her2st then cSCC in one run'
    )
    parser.add_argument(
        '--all_folds',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Train all folds (iterate over all slices) in one run（her2st: 32 folds）'
    )
    parser.add_argument(
        '--num_folds',
        type=int,
        default=0,
        help='Override number of folds when using --all_folds (0 = auto)'
    )

    parser.add_argument('--dim_in', type=int, default=1024)
    parser.add_argument('--dim_hidden', type=int, default=1024)
    parser.add_argument('--dim_out', type=int, default=785)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--decoder_layer', type=int, default=3)
    parser.add_argument('--decoder_head', type=int, default=2)

    parser.add_argument(
        '--mask_rate',
        type=float,
        default=0.1,
        help='Masking ratio for Context Reconstruction（cSCC 由 DATASET_DEFAULTS 覆盖为 0.4，见 §9.1.6）',
    )
    parser.add_argument('--mask_block_size', type=int, default=8, help='Approx block size for spatial masking')
    parser.add_argument('--wikg_top', type=int, default=8, help='Top K for physical/semantic GNN graphs')
    parser.add_argument('--tau', type=float, default=0.15, help='Temperature for Soft Contrastive Loss（context 对比 logits）')
    parser.add_argument('--dist_lambda', type=float, default=0.1, help='Lambda for exp(-d/lambda) distance decay（her2st 由 DATASET_DEFAULTS 覆盖为 0.2）')
    parser.add_argument('--gate_entropy_weight', type=float, default=0.0,
    help='Encourage non-collapsed gating via entropy regularization')

    parser.add_argument('--dist_sigma', type=float, default=0.3, help='Sigma for distance kernel exp(-d/sigma) in Soft-Con targets')
    parser.add_argument('--soft_target_eps', type=float, default=3e-5, help='Epsilon smoothing for soft target rows')
    parser.add_argument(
        '--attn_k',
        type=int,
        default=24,
        help='Transformer 局部注意力 kNN 窗口（0=全局）。未显式传入时由 DATASET_DEFAULTS 覆盖：her2st=16，cSCC=24。',
    )
    parser.add_argument(
        '--disable_ds_gnn',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='消融：跳过 DS-GNN，直接用 Transformer context 特征预测',
    )
    parser.add_argument(
        '--disable_coord_emb',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='消融：去掉 x/y 坐标 embedding（i_main 仅保留图像特征）',
    )

    parser.add_argument('--ema_decay', type=float, default=0.998, help='EMA decay for gene teacher encoder')

    parser.add_argument(
        '--use_ema_target',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Use EMA teacher gene encoder as fixed target for reconstruction'
    )

    parser.add_argument(
        '--w_softcon',
        type=float,
        default=None,
        help='Soft-Con 权重；',
    )
    parser.add_argument(
        '--w_recon',
        type=float,
        default=None,
        help='Mask-Recon 权重；',
    )
    parser.add_argument('--w_mse', type=float, default=1.0)

    parser.add_argument('--w_zinb', type=float, default=0.25)

    parser.add_argument(
        '--use_cache_matrices',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Cache dist_norm (及 morph_sim，供数据管线兼容) per sample to disk'
    )
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to store cached matrices (default: <output_dir>/cache)')

    parser.add_argument(
        '--stop_on_nan',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Stop training immediately when NaN/Inf is detected (recommended on server)'
    )

    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument("--fold", type=int, default=0, help="fold number")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='AdamW 学习率。parser 默认 1e-4；cSCC/her2st 可由 apply_dataset_defaults 覆盖（除非命令行显式 --lr）。',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='全局随机种子基值；未设置则保持非确定性行为。实际种子=seed+fold*FOLD_SEED_STRIDE（见 utils.effective_train_seed）',
    )

    args_cmd, _ = parser.parse_known_args()
    return args_cmd
