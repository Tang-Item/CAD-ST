import numpy as np
import os
import torch
import pandas as pd
import scanpy as sc
import scprep as scp
from PIL import ImageFile, Image
from collections import defaultdict
import glob
import random
from torch.utils.data import Dataset
import hashlib
import warnings

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

_CACHE_VERSION = 1

def _compute_dist_norm_np(pos_xy: np.ndarray) -> np.ndarray:
    pos = pos_xy.astype(np.float32)
    sq = (pos ** 2).sum(axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (pos @ pos.T)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2 + 1e-8)
    return dist / (dist.max() + 1e-8)

def _compute_morph_sim_np(img_f: np.ndarray) -> np.ndarray:
    x = img_f.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return x @ x.T

def _sha1_bytes(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()

def _build_cache_signature(positions_xy: np.ndarray, img_path: str) -> dict:
    pos_hash = _sha1_bytes(positions_xy.astype(np.int32).tobytes())
    try:
        st = os.stat(img_path)
        img_mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        img_size = int(st.st_size)
    except Exception:
        img_mtime_ns = -1
        img_size = -1
    return {
        "version": _CACHE_VERSION,
        "pos_hash": pos_hash,
        "img_mtime_ns": img_mtime_ns,
        "img_size": img_size,
    }

def _signature_matches(sig: dict, positions_xy: np.ndarray, img_path: str) -> bool:
    if not isinstance(sig, dict):
        return False
    if int(sig.get("version", -1)) != _CACHE_VERSION:
        return False
    cur = _build_cache_signature(positions_xy, img_path)
    return (
        sig.get("pos_hash") == cur.get("pos_hash")
        and int(sig.get("img_mtime_ns", -2)) == int(cur.get("img_mtime_ns", -1))
        and int(sig.get("img_size", -2)) == int(cur.get("img_size", -1))
    )

def apply_pos_mode(loc: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return loc
    if mode not in ("minmax511", "minmax63"):
        raise ValueError(f"Unknown pos_mode: {mode}")
    hi = 511.0 if mode == "minmax511" else 63.0
    xy = np.asarray(loc, dtype=np.float32)
    out = np.zeros_like(xy, dtype=np.float64)
    for j in range(xy.shape[1]):
        lo = float(xy[:, j].min())
        mx = float(xy[:, j].max())
        if mx - lo < 1e-8:
            out[:, j] = hi / 2.0
        else:
            out[:, j] = (xy[:, j] - lo) / (mx - lo) * hi
    out = np.round(out).clip(0.0, hi)
    return out.astype(np.int64)

def _load_or_create_cache_npz(cache_path: str, positions_xy: np.ndarray, img_path: str, img_f: np.ndarray):
    if os.path.exists(cache_path):
        try:
            d = np.load(cache_path)

            sig = None
            if "sig_version" in d.files:
                sig = {
                    "version": int(d["sig_version"]),
                    "pos_hash": str(d["sig_pos_hash"]),
                    "img_mtime_ns": int(d["sig_img_mtime_ns"]),
                    "img_size": int(d["sig_img_size"]),
                }
            if sig is not None and _signature_matches(sig, positions_xy, img_path):
                return d["dist_norm"].astype(np.float32), d["morph_sim"].astype(np.float32)
        except Exception:
            pass
    dist_norm = _compute_dist_norm_np(positions_xy)
    morph_sim = _compute_morph_sim_np(img_f)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sig = _build_cache_signature(positions_xy, img_path)
    np.savez_compressed(
        cache_path,
        dist_norm=dist_norm.astype(np.float16),
        morph_sim=morph_sim.astype(np.float16),
        sig_version=np.int32(sig["version"]),
        sig_pos_hash=np.array(sig["pos_hash"]),
        sig_img_mtime_ns=np.int64(sig["img_mtime_ns"]),
        sig_img_size=np.int64(sig["img_size"]),
    )
    return dist_norm.astype(np.float32), morph_sim.astype(np.float32)

def split_list(seed=None):
    if seed is not None:
        random.seed(seed)
    data = list(range(31))
    train_size = len(data) * 8 // 9
    test_size = len(data) - train_size
    train_set = random.sample(data, train_size)
    test_set =[item for item in data if item not in train_set]
    return train_set, test_set

class ViT_HER2ST(Dataset):
    def __init__(self, train=True, fold=1, r=4, flatten=True, ori=False, adj=False, prune='Grid', neighs=4, val=True,
                 use_cache_matrices: bool = False, cache_dir: str = "data/cache_mats",
                 pos_mode: str = "raw", require_spot_ids: bool = False):
        super(ViT_HER2ST, self).__init__()
        self.dir = "data/her2st"
        self.cnt_dir = f"{self.dir}/ST-cnts"
        self.img_dir = f"{self.dir}/ST-imgs"

        pos_dir_candidates = [f"{self.dir}/ST-profiles", f"{self.dir}/ST-spotfiles"]
        self.pos_dir = next((p for p in pos_dir_candidates if os.path.exists(p)), pos_dir_candidates[0])
        self.lbl_dir = f"{self.dir}/ST-pat/lbl"
        self.r = 224 // 2
        gene_list = list(np.load(f"data/her2st/her_hvg_cut_1000.npy", allow_pickle=True))

        self.gene_list = gene_list
        names = self._discover_her2st_samples()
        self.train = train
        self.val = val
        self.ori = ori
        self.adj = adj
        samples = names[1:33]

        te_names = [samples[fold]]
        print("test sample:", te_names)
        tr_names = list(set(samples) - set(te_names))

        if self.train:
            self.names = tr_names
        else:
            self.names = te_names

        self.names.sort()

        self.img_dict = None
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}
        self.label = {i: None for i in self.names}
        self.lbl2id = {
            'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
            'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
        }
        if not self.train and self.names[0] in['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values
            self.label[self.names[0]] = lbl
        elif self.train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    lbl = torch.Tensor([self.lbl2id[i] for i in lbl])
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            for i, m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf
        self.center_dict = {
            i: np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x','y']].values for i, m in self.meta_dict.items()}
        self.patch_dict = defaultdict(type(None))
        self.lengths =[len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

        env_embed_dir = os.getenv("CADST_HER2ST_EMBED_DIR", "").strip()
        if env_embed_dir:
            self.patch_path = env_embed_dir.rstrip("/\\") + os.sep
        else:
            aligned_dir = os.path.join(self.dir, "phikonv2_embedding_aligned")
            legacy_dir = os.path.join(self.dir, "phikonv2_embedding")
            self.patch_path = (aligned_dir if os.path.exists(aligned_dir) else legacy_dir) + os.sep
        self._spot_id_order_missing_warned = set()
        env_disable = os.getenv("CADST_ALLOW_UNVERIFIED_EMBEDDING", "").strip()
        allow_unverified = env_disable in ("1", "true", "TRUE", "yes", "YES")
        env_require = os.getenv("CADST_REQUIRE_SPOT_IDS", "").strip()
        force_strict_env = env_require in ("1", "true", "TRUE", "yes", "YES")

        if allow_unverified:
            self.require_spot_ids = bool(require_spot_ids) or force_strict_env
        else:
            self.require_spot_ids = True
        self.use_cache_matrices = use_cache_matrices
        self.cache_dir = cache_dir
        self.pos_mode = pos_mode

    def _discover_her2st_samples(self):
        names = []
        for fn in os.listdir(self.cnt_dir):
            if fn.endswith(".tsv.gz"):
                sample = fn[:-7]
            elif fn.endswith(".tsv"):
                sample = fn[:-4]
            else:
                continue
            names.append(sample)
        names = sorted(set(names))
        if len(names) < 33:
            raise RuntimeError(f"Unexpected HER2ST count files: only {len(names)} samples found in {self.cnt_dir}")
        return names

    def _maybe_align_img_features(self, ID, img_f):
        spot_id_path = os.path.join(self.patch_path, f"{ID}_spot_ids.txt")
        if not os.path.exists(spot_id_path):
            msg = (
                f"[align-check] Missing spot id order file for {ID}: {spot_id_path}. "
                "Cannot verify embedding row order against meta index."
            )
            if self.require_spot_ids:
                raise FileNotFoundError(msg + " Refuse to proceed because strict alignment mode is enabled.")
            if ID not in self._spot_id_order_missing_warned:
                warnings.warn(msg, RuntimeWarning)
                self._spot_id_order_missing_warned.add(ID)
            return img_f

        spot_ids = pd.read_csv(spot_id_path, header=None)[0].astype(str).tolist()
        meta_ids = self.meta_dict[ID].index.astype(str).tolist()
        if len(spot_ids) != len(meta_ids):
            raise AssertionError(
                f"[align-check] Spot id length mismatch for {ID}: "
                f"embedding ids={len(spot_ids)}, meta ids={len(meta_ids)}"
            )
        if set(spot_ids) != set(meta_ids):
            raise AssertionError(f"[align-check] Spot id set mismatch for {ID}")

        spot_idx = {sid: i for i, sid in enumerate(spot_ids)}
        order = [spot_idx[sid] for sid in meta_ids]
        return img_f[order]

    def __getitem__(self, index):
        ID = self.id2name[index]
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        loc = apply_pos_mode(np.asarray(loc), self.pos_mode)

        positions = torch.LongTensor(loc)
        img_path = f"{self.patch_path}{ID}.npy"
        img_f = np.load(img_path)
        img_f = self._maybe_align_img_features(ID, img_f)
        assert img_f.shape[0] == len(exps), (ID, img_f.shape, np.asarray(exps).shape)
        assert img_f.shape[0] == len(positions), (ID, img_f.shape, positions.shape)
        assert np.isfinite(img_f).all(), f"[align-check] Non-finite image embedding in {ID}"

        dist_norm = None
        morph_sim = None
        if self.use_cache_matrices:
            cache_path = os.path.join(self.cache_dir, "her2st", f"{ID}.npz")
            dist_norm, morph_sim = _load_or_create_cache_npz(cache_path, positions_xy=loc, img_path=img_path, img_f=img_f)
            dist_norm = torch.from_numpy(dist_norm)
            morph_sim = torch.from_numpy(morph_sim)
        else:
            dist_norm = torch.empty(0)
            morph_sim = torch.empty(0)

        data = [torch.Tensor(exps), torch.Tensor(img_f), positions, dist_norm, morph_sim, torch.Tensor(centers)]
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        return data

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path_tsv = os.path.join(self.cnt_dir, f"{name}.tsv")
        path_gz = os.path.join(self.cnt_dir, f"{name}.tsv.gz")
        if os.path.exists(path_tsv):
            path = path_tsv
        elif os.path.exists(path_gz):
            path = path_gz
        else:
            raise FileNotFoundError(f"Count file not found: {path_tsv} or {path_gz}")

        df = pd.read_csv(path, sep='\t', index_col=0, compression='infer')
        return df

    def get_pos(self, name):
        path = os.path.join(self.pos_dir, f"{name}_selection.tsv")
        if not os.path.exists(path):
            alt_dir = f"{self.dir}/ST-spotfiles" if "ST-profiles" in self.pos_dir else f"{self.dir}/ST-profiles"
            alt_path = os.path.join(alt_dir, f"{name}_selection.tsv")
            if os.path.exists(alt_path):
                path = alt_path
        df = pd.read_csv(path, sep='\t')

        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['x', 'y']).copy()
        x = np.around(df['x'].values).astype(int)
        y = np.around(df['y'].values).astype(int)
        id =[]
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index("id"), how="inner")
        coord_cols = ["x", "y", "pixel_x", "pixel_y"]
        if not meta[coord_cols].notna().all().all():
            raise AssertionError(f"[meta-check] NaN coords after join for {name}")
        if meta.index.duplicated().any():
            raise AssertionError(f"[meta-check] Duplicate spot ids after join for {name}")
        if len(meta) == 0:
            raise AssertionError(f"[meta-check] Empty meta after join for {name}")
        return meta

    def get_lbl(self, name):
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')

        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['x', 'y']).copy()
        x = np.around(df['x'].values).astype(int)
        y = np.around(df['y'].values).astype(int)
        id =[]
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id', inplace=True)
        return df

class ViT_SKIN(Dataset):
    def __init__(self, train=True, r=2, norm=False, fold=0, flatten=True, ori=False, adj=False, prune='NA', neighs=4,
                 use_cache_matrices: bool = False, cache_dir: str = "data/cache_mats", require_spot_ids: bool = False):
        super(ViT_SKIN, self).__init__()
        self.dir = "data/cSCC/GSE144240_RAW"
        self.embed_root = "data/cSCC"
        self.r = 224 // r
        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names =[]
        for i in patients:
            for j in reps:
                names.append(i + '_ST_' + j)
        gene_list = list(np.load(f'data/cSCC/skin_hvg_cut_1000.npy', allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        self.names.sort()

        self.img_dict = None
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
                for i, m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
                for i, m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf
        self.center_dict = {
            i: np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x','y']].values for i, m in self.meta_dict.items()}
        self.patch_dict = defaultdict(type(None))
        self.lengths =[len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

        env_embed_dir = os.getenv("CADST_CSCC_EMBED_DIR", "").strip()
        if env_embed_dir:
            self.patch_path = env_embed_dir.rstrip("/\\") + os.sep
        else:
            aligned_dir = os.path.join(self.embed_root, "phikonv2_embedding_aligned")
            legacy_dir = os.path.join(self.embed_root, "phikonv2_embedding")
            self.patch_path = (aligned_dir if os.path.exists(aligned_dir) else legacy_dir) + os.sep
        print(f"cSCC Phikon embedding dir: {self.patch_path}")
        self._spot_id_order_missing_warned = set()
        env_disable = os.getenv("CADST_ALLOW_UNVERIFIED_EMBEDDING", "").strip()
        allow_unverified = env_disable in ("1", "true", "TRUE", "yes", "YES")
        env_require = os.getenv("CADST_REQUIRE_SPOT_IDS", "").strip()
        force_strict_env = env_require in ("1", "true", "TRUE", "yes", "YES")
        if allow_unverified:
            self.require_spot_ids = bool(require_spot_ids) or force_strict_env
        else:
            self.require_spot_ids = True
        self.use_cache_matrices = use_cache_matrices
        self.cache_dir = cache_dir

    def _maybe_align_img_features(self, ID, img_f):
        spot_id_path = os.path.join(self.patch_path, f"{ID}_spot_ids.txt")
        if not os.path.exists(spot_id_path):
            msg = (
                f"[align-check] Missing spot id order file for {ID}: {spot_id_path}. "
                "Cannot verify embedding row order against meta index."
            )
            if self.require_spot_ids:
                raise FileNotFoundError(msg + " Refuse to proceed because strict alignment mode is enabled.")
            if ID not in self._spot_id_order_missing_warned:
                warnings.warn(msg, RuntimeWarning)
                self._spot_id_order_missing_warned.add(ID)
            return img_f

        spot_ids = pd.read_csv(spot_id_path, header=None)[0].astype(str).tolist()
        meta_ids = self.meta_dict[ID].index.astype(str).tolist()
        if len(spot_ids) != len(meta_ids):
            raise AssertionError(
                f"[align-check] Spot id length mismatch for {ID}: "
                f"embedding ids={len(spot_ids)}, meta ids={len(meta_ids)}"
            )
        if set(spot_ids) != set(meta_ids):
            raise AssertionError(f"[align-check] Spot id set mismatch for {ID}")

        spot_idx = {sid: i for i, sid in enumerate(spot_ids)}
        order = [spot_idx[sid] for sid in meta_ids]
        return img_f[order]

    def __getitem__(self, index):
        ID = self.id2name[index]
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        positions = torch.LongTensor(loc)
        img_path = f'{self.patch_path}{ID}.npy'
        img_f = np.load(img_path)
        img_f = self._maybe_align_img_features(ID, img_f)
        assert img_f.shape[0] == len(exps), (ID, img_f.shape, np.asarray(exps).shape)
        assert img_f.shape[0] == len(positions), (ID, img_f.shape, positions.shape)
        assert np.isfinite(img_f).all(), f"[align-check] Non-finite image embedding in {ID}"

        dist_norm = None
        morph_sim = None
        if self.use_cache_matrices:
            cache_path = os.path.join(self.cache_dir, "cSCC", f"{ID}.npz")
            dist_norm, morph_sim = _load_or_create_cache_npz(cache_path, positions_xy=loc, img_path=img_path, img_f=img_f)
            dist_norm = torch.from_numpy(dist_norm)
            morph_sim = torch.from_numpy(morph_sim)
        else:
            dist_norm = torch.empty(0)
            morph_sim = torch.empty(0)

        data =[torch.Tensor(exps), torch.Tensor(img_f), positions, dist_norm, morph_sim, torch.Tensor(centers)]
        if self.ori:
            data +=[torch.Tensor(oris), torch.Tensor(sfs)]
        return data

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pattern = os.path.join(self.dir, f"*{name}*.jpg")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"Image file not found for {name}. Pattern: {pattern}")
        im = Image.open(matches[0])
        return im

    def get_cnt(self, name):
        pattern = os.path.join(self.dir, f"*{name}*_stdata.tsv")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"Count file not found for {name}. Pattern: {pattern}")
        df = pd.read_csv(matches[0], sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        pattern = os.path.join(self.dir, f"*spot*{name}*.tsv")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"Spot file not found for {name}. Pattern: {pattern}")
        df = pd.read_csv(matches[0], sep='\t')
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['x', 'y']).copy()
        x = np.around(df['x'].values).astype(int)
        y = np.around(df['y'].values).astype(int)
        id =[]
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df

    def get_meta(self, name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'), how='inner')
        return meta
