import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import anndata as ann
import math
import copy
from torch_geometric.nn import GATConv, knn_graph

from utils import get_R

class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.linear_2(x)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.2):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.dropout = dropout
        self.d_k = dim // heads
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None, dropout=0.2):
        d_k = query.shape[-1]
        attention_weights = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        attention_weights = attention_weights.softmax(dim=-1)
        if dropout > 0:
            attention_weights = self.dropout_layer(attention_weights)
        return (attention_weights @ value), attention_weights

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q).view(q.shape[0], self.heads, self.d_k).transpose(0, 1)
        key = self.w_k(k).view(k.shape[0], self.heads, self.d_k).transpose(0, 1)
        value = self.w_v(v).view(v.shape[0], self.heads, self.d_k).transpose(0, 1)

        x, self.attention_weights = self.attention(query, key, value, mask, dropout=self.dropout)
        x = x.contiguous().view(x.shape[1], -1)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.2):
        super().__init__()
        self.attention_block = MultiHeadAttentionBlock(dim, heads, dropout)
        self.feed_forward_block = FeedForwardBlock(dim, mlp_dim, dropout)
        self.residual_block = nn.ModuleList([ResidualConnection(dim, dropout) for _ in range(2)])

    def forward(self, x, mask=None):
        x = self.residual_block[0](x, lambda x: self.attention_block(x, x, x, mask))
        x = self.residual_block[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, dim, layers, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(dim, heads, mlp_dim, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LR_Scheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr

class MeanAct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

def pairwise_dist_norm(pos: torch.Tensor) -> torch.Tensor:
    pos = pos.float()

    sq = (pos ** 2).sum(dim=1, keepdim=True)
    dist2 = sq + sq.T - 2.0 * (pos @ pos.T)
    dist2 = torch.clamp(dist2, min=0.0)
    dist = torch.sqrt(dist2 + 1e-8)
    return dist / (dist.max() + 1e-8)

def row_normalize(mat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = mat.sum(dim=1, keepdim=True).clamp_min(eps)
    return mat / denom

def ZINB_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
    eps = 1e-10
    if isinstance(scale_factor, float):
        scale_factor = np.full((len(mean),), scale_factor)
    scale_factor = scale_factor[:, None]
    mean = mean * scale_factor

    t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
    t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0-pi+eps)
    zero_nb = torch.pow(disp/(disp+mean+eps), disp)
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    if ridge_lambda > 0:
        ridge = ridge_lambda*torch.square(pi)
        result += ridge
    result = torch.mean(result)
    return result

class DSGNN(nn.Module):
    def __init__(self, dim, k=6):
        super().__init__()
        self.k = k
        self.phys_conv = GATConv(dim, dim // 2, heads=2, concat=True)
        self.sem_conv = GATConv(dim, dim // 2, heads=2, concat=True)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, pos, dist_norm, dist_lambda: float = 0.3):
        edge_index_phys = knn_graph(pos.float(), k=self.k, loop=True)
        h_phys = self.act(self.phys_conv(x, edge_index_phys))

        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        sim = x_norm @ x_norm.t()

        decay = torch.exp(-dist_norm / max(float(dist_lambda), 1e-6))
        sem_score = sim * decay

        _, topk_idx = torch.topk(sem_score, k=self.k, dim=1)
        row = torch.arange(x.size(0), device=x.device).repeat_interleave(self.k)
        col = topk_idx.flatten()
        edge_index_sem = torch.stack([row, col], dim=0)

        h_sem = self.act(self.sem_conv(x, edge_index_sem))

        h_cat = torch.cat([h_phys, h_sem], dim=-1)
        g = self.gate(h_cat)
        h_fused = g * h_phys + (1 - g) * h_sem

        return self.norm(x + h_fused), g

class CAD_ST(pl.LightningModule):
    def __init__(self, args):
        super(CAD_ST, self).__init__()
        self.args = args
        dim_in = args.dim_in
        dim_hidden = args.dim_hidden
        dim_out = args.dim_out
        dropout = args.dropout

        self.image_encoder = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_in, dim_hidden)
        )
        self.gene_encoder = nn.Sequential(
            nn.Linear(dim_out, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, dim_hidden)
        )

        self.use_ema_target = bool(getattr(args, 'use_ema_target', False))
        self.ema_decay = float(getattr(args, 'ema_decay', 0.996))
        if self.use_ema_target:
            self.gene_encoder_ema = copy.deepcopy(self.gene_encoder)
            for p in self.gene_encoder_ema.parameters():
                p.requires_grad = False
        else:
            self.gene_encoder_ema = None

        self.mask_token = nn.Parameter(torch.randn(1, dim_hidden))
        self.mask_rate = args.mask_rate if hasattr(args, 'mask_rate') else 0.25

        self.embed_x = nn.Embedding(512, dim_hidden)
        self.embed_y = nn.Embedding(512, dim_hidden)

        self.context_encoder = Encoder(dim=dim_hidden, layers=args.decoder_layer, heads=args.decoder_head, mlp_dim=1024, dropout=dropout)

        self.ds_gnn = DSGNN(dim=dim_hidden, k=args.wikg_top)

        self.gene_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

        self.mean = nn.Sequential(nn.Linear(dim_hidden, dim_out), MeanAct())
        self.disp = nn.Sequential(nn.Linear(dim_hidden, dim_out), DispAct())
        self.pi = nn.Sequential(nn.Linear(dim_hidden, dim_out), nn.Sigmoid())

        self.lr_scheduler = None

    def build_soft_targets(self, dist_norm: torch.Tensor) -> torch.Tensor:
        sigma = float(getattr(self.args, 'dist_sigma', 0.25))
        eps = float(getattr(self.args, 'soft_target_eps', 1e-4))
        dist_kernel = torch.exp(-dist_norm / max(sigma, 1e-6))
        joint = dist_kernel + eps
        n = joint.shape[0]
        idx = torch.arange(n, device=joint.device, dtype=torch.long)
        joint = joint.clone()
        joint[idx, idx] = 1.0 + eps
        return row_normalize(joint)

    @torch.no_grad()
    def _update_ema_teacher(self):
        if not self.use_ema_target or self.gene_encoder_ema is None:
            return
        d = float(self.ema_decay)
        for p_ema, p in zip(self.gene_encoder_ema.parameters(), self.gene_encoder.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=(1.0 - d))

    def _spatial_block_mask(self, pos: torch.Tensor, num_mask: int, block_size: int) -> torch.Tensor:
        N = pos.size(0)
        num_mask = int(min(max(num_mask, 0), N))
        if num_mask == 0:
            return torch.zeros(N, dtype=torch.bool, device=pos.device)
        block_size = int(max(block_size, 1))

        mask = torch.zeros(N, dtype=torch.bool, device=pos.device)

        dist = pairwise_dist_norm(pos)
        remaining = num_mask
        while remaining > 0:
            candidates = (~mask).nonzero(as_tuple=False).flatten()
            if candidates.numel() == 0:
                break
            seed = candidates[torch.randint(0, candidates.numel(), (1,), device=pos.device)].item()
            k = min(block_size, remaining)
            nn_idx = torch.topk(-dist[seed], k=k, largest=True).indices
            nn_idx = nn_idx[~mask[nn_idx]]
            take = nn_idx[:k]
            mask[take] = True
            remaining = num_mask - int(mask.sum().item())
        return mask

    def forward(self, gene, image, pos, dist_norm: torch.Tensor | None = None):
        i_f = self.image_encoder(image)
        g_f = self.gene_encoder(gene)

        g_f_norm = g_f / (g_f.norm(dim=1, keepdim=True) + 1e-8)

        if dist_norm is None or dist_norm.numel() == 0:
            dist_norm = pairwise_dist_norm(pos)
        else:
            dist_norm = dist_norm.to(pos.device, dtype=torch.float32)

        x_emb = self.embed_x(pos[:, 0].long().clamp(0, 511))
        y_emb = self.embed_y(pos[:, 1].long().clamp(0, 511))

        tau = float(getattr(self.args, 'tau', 0.2))
        w_soft = float(getattr(self.args, 'w_softcon', 0.5))
        soft_con_loss = torch.zeros((), device=i_f.device, dtype=i_f.dtype)

        def _softcon_from_context_norm(context_norm: torch.Tensor) -> torch.Tensor:
            device_type = 'cuda' if i_f.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=False):
                img32 = context_norm.float()
                g32 = g_f_norm.float()
                soft_target = self.build_soft_targets(dist_norm.float()).float()
                logits_img2gene = (img32 @ g32.T) / float(tau)
                logits_gene2img = (g32 @ img32.T) / float(tau)
                return -(
                    (soft_target * F.log_softmax(logits_img2gene, dim=1)).sum(dim=1).mean()
                    + (soft_target.T * F.log_softmax(logits_gene2img, dim=1)).sum(dim=1).mean()
                ) / 2.0

        mask = None

        w_rec = float(getattr(self.args, 'w_recon', 1.0))
        if self.training and w_rec != 0.0:
            N = i_f.shape[0]
            num_mask = int(N * self.mask_rate)
            block_size = int(getattr(self.args, 'mask_block_size', 32))
            mask = self._spatial_block_mask(pos, num_mask=num_mask, block_size=block_size)

        attn_k = int(getattr(self.args, 'attn_k', 0))
        attn_mask = None
        if attn_k > 0:
            n_spot = dist_norm.size(0)
            k = min(max(attn_k, 1), n_spot)
            knn_idx = torch.topk(dist_norm, k=k, dim=1, largest=False).indices
            attn_mask = torch.zeros((n_spot, n_spot), device=dist_norm.device, dtype=torch.bool)
            row = torch.arange(n_spot, device=dist_norm.device).unsqueeze(1).expand(-1, k)
            attn_mask[row, knn_idx] = True

            attn_mask.fill_diagonal_(True)

        i_main_img = i_f.clone()
        if self.training and w_rec != 0.0 and mask is not None and mask.sum() > 0:
            i_main_img[mask] = self.mask_token.to(dtype=i_main_img.dtype, device=i_main_img.device)
        if bool(getattr(self.args, 'disable_coord_emb', False)):
            i_main = i_main_img
        else:
            i_main = i_main_img + x_emb + y_emb
        context_main = self.context_encoder(i_main, mask=attn_mask)

        if w_soft != 0.0:
            context_norm = context_main / (context_main.norm(dim=1, keepdim=True) + 1e-8)
            soft_con_loss = _softcon_from_context_norm(context_norm)

        if self.training and w_rec != 0.0 and mask is not None and mask.sum() > 0:
            if self.use_ema_target and self.gene_encoder_ema is not None:
                with torch.no_grad():
                    target_latent = self.gene_encoder_ema(gene)
            else:
                target_latent = g_f.detach()
            recon_loss = F.mse_loss(context_main[mask], target_latent.detach()[mask])
        else:
            recon_loss = torch.tensor(0.0, device=context_main.device)

        if bool(getattr(self.args, 'disable_ds_gnn', False)):
            refined_f = context_main
            gate = None
        else:
            dist_lambda = float(getattr(self.args, "dist_lambda", 0.3))
            refined_f, gate = self.ds_gnn(context_main, pos, dist_norm, dist_lambda=dist_lambda)

        pred_g = self.gene_head(refined_f)
        m = self.mean(refined_f)
        d = self.disp(refined_f)
        p = self.pi(refined_f)
        extra = (m, d, p)

        return pred_g, extra, soft_con_loss, recon_loss, gate

    def training_step(self, batch, batch_idx):
        w_soft = float(getattr(self.args, 'w_softcon', 0.5))
        w_rec = float(getattr(self.args, 'w_recon', 1.0))
        w_mse = float(getattr(self.args, 'w_mse', 1.0))
        w_zinb = float(getattr(self.args, 'w_zinb', 0.25))
        mse_only = (w_mse != 0.0) and (w_soft == 0.0) and (w_rec == 0.0) and (w_zinb == 0.0)

        if len(batch) >= 8:
            g, i, pos, dist_norm, _morph_sim, _, oris, sfs = batch
        else:
            g, i, pos, _, oris, sfs = batch
            dist_norm = None
        g = g.squeeze(0)
        i = i.squeeze(0)
        pos = pos.squeeze(0)
        if dist_norm is not None:
            dist_norm = dist_norm.squeeze(0)

        if mse_only:
            device_type = 'cuda' if g.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=False):
                pred_g, extra, soft_con_loss, recon_loss, gate = self.forward(
                    g.float(), i.float(), pos, dist_norm=dist_norm
                )
                mse_loss = F.mse_loss(pred_g, g.float())

            zinb_loss = torch.zeros((), device=g.device, dtype=torch.float32)
            m = d = p = None
        else:
            pred_g, extra, soft_con_loss, recon_loss, gate = self.forward(
                g, i, pos, dist_norm=dist_norm
            )
            m, d, p = extra

            mse_loss = F.mse_loss(pred_g.float(), g.float())
            zinb_loss = ZINB_loss(
                oris.squeeze(0).float(),
                m.float(),
                d.float(),
                p.float(),
                sfs.squeeze(0).float(),
            )

        if not torch.isfinite(g).all():
            self.log('nan_in_g', 1.0, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if not torch.isfinite(pred_g).all():
            self.log('nan_in_pred_g', 1.0, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        def _is_bad(x: torch.Tensor) -> torch.Tensor:
            return (~torch.isfinite(x)).to(dtype=torch.float32)

        nan_soft = _is_bad(soft_con_loss)
        nan_recon = _is_bad(recon_loss)
        nan_mse = _is_bad(mse_loss)
        nan_zinb = _is_bad(zinb_loss)
        self.log('nan_softcon', nan_soft, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('nan_recon', nan_recon, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('nan_mse', nan_mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('nan_zinb', nan_zinb, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        def _wmul(w: float, loss: torch.Tensor) -> torch.Tensor:
            if w == 0.0:
                return torch.zeros((), device=loss.device, dtype=loss.dtype)
            return loss * w

        train_loss = _wmul(w_soft, soft_con_loss) + _wmul(w_rec, recon_loss) + _wmul(w_mse, mse_loss) + _wmul(w_zinb, zinb_loss)

        gate_w = float(getattr(self.args, 'gate_entropy_weight', 0.01))

        if mse_only:
            gate_w = 0.0
        gate_entropy = None
        if gate is not None and gate_w > 0:
            eps = 1e-6
            gate32 = gate.float()
            gate_clamped = gate32.clamp(eps, 1 - eps)
            gate_entropy = -(gate_clamped * torch.log(gate_clamped) + (1 - gate_clamped) * torch.log(1 - gate_clamped)).mean()
            train_loss = train_loss.float() - float(gate_w) * gate_entropy
            self.log('gate_entropy', gate_entropy, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log('gate_mean', gate.mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            gate_entropy = torch.zeros((), device=train_loss.device, dtype=torch.float32)

        nan_gate = _is_bad(gate_entropy)
        self.log('nan_gate_entropy', nan_gate, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        if not torch.isfinite(train_loss):
            self.log('nan_train_loss', 1.0, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            if bool(getattr(self.args, 'stop_on_nan', False)):
                bad_soft = bool(nan_soft.item()) and (w_soft != 0.0)
                bad_recon = bool(nan_recon.item()) and (w_rec != 0.0)
                bad_mse = bool(nan_mse.item()) and (w_mse != 0.0)
                bad_zinb = bool(nan_zinb.item()) and (w_zinb != 0.0)
                bad_gate = bool(nan_gate.item()) and (gate_w != 0.0)
                raise RuntimeError(
                    f"NaN/Inf detected. "
                    f"soft={bad_soft}, recon={bad_recon}, mse={bad_mse}, zinb={bad_zinb}, gate_entropy={bad_gate} "
                    f"(weights: soft={w_soft}, recon={w_rec}, mse={w_mse}, zinb={w_zinb}, gate_entropy={gate_w})"
                )
            train_loss = (w_mse * mse_loss)

        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('soft_con_loss', soft_con_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('zinb_loss', zinb_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        if self.lr_scheduler:
            self.lr_scheduler.step()
        return train_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_ema_teacher()

    def validation_step(self, batch, batch_idx):
        if len(batch) >= 8:
            g, i, pos, dist_norm, _morph_sim, _, _, _ = batch
        else:
            g, i, pos, _, _, _ = batch
            dist_norm = None
        g = g.squeeze(0)
        i = i.squeeze(0)
        pos = pos.squeeze(0)
        if dist_norm is not None:
            dist_norm = dist_norm.squeeze(0)

        pred_g, _, _, _, _ = self.forward(g, i, pos, dist_norm=dist_norm)
        pred_np = pred_g.detach().cpu().numpy()
        true_np = g.detach().cpu().numpy()
        p, r = get_R(pred_np, true_np)
        pcc = np.nanmean(p)
        valid_gene_frac = float(np.isfinite(p).mean())
        const_pred_frac = float((pred_np.std(axis=0) < 1e-12).mean())
        const_true_frac = float((true_np.std(axis=0) < 1e-12).mean())
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)
        self.log('valid_gene_frac', valid_gene_frac, prog_bar=False, sync_dist=True)
        self.log('const_pred_frac', const_pred_frac, prog_bar=False, sync_dist=True)
        self.log('const_true_frac', const_true_frac, prog_bar=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        if len(batch) >= 8:
            g, i, pos, dist_norm, _morph_sim, centers, _, _ = batch
        else:
            g, i, pos, centers, _, _ = batch
            dist_norm = None
        g = g.squeeze(0)
        i = i.squeeze(0)
        pos = pos.squeeze(0)
        centers = centers.squeeze(0)
        if dist_norm is not None:
            dist_norm = dist_norm.squeeze(0)

        pred_g, _, _, _, _ = self.forward(g, i, pos, dist_norm=dist_norm)

        pred_np = pred_g.detach().cpu().numpy()
        true_np = g.detach().cpu().numpy()
        adata = ann.AnnData(X=pred_np)
        adata.obsm["spatial"] = centers.detach().cpu().numpy()
        p, r = get_R(pred_np, true_np)
        pcc = np.nanmean(p)
        valid_gene_frac = float(np.isfinite(p).mean())
        const_pred_frac = float((pred_np.std(axis=0) < 1e-12).mean())
        const_true_frac = float((true_np.std(axis=0) < 1e-12).mean())

        self.log('pcc', pcc, prog_bar=True, sync_dist=True)
        self.log('valid_gene_frac', valid_gene_frac, prog_bar=False, sync_dist=True)
        self.log('const_pred_frac', const_pred_frac, prog_bar=False, sync_dist=True)
        self.log('const_true_frac', const_true_frac, prog_bar=False, sync_dist=True)
        self.p = p
        self.r = r
        self.data = adata

        self.pred_np = pred_np
        self.true_np = true_np
        self.centers_np = centers.detach().cpu().numpy()

    def configure_optimizers(self):
        lr = float(getattr(self.args, "lr", 1e-4))
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

        iter_per_epoch = int(getattr(self.args, "iter_per_epoch", 0) or 0)
        if iter_per_epoch <= 0 and getattr(self, "trainer", None) is not None:
            try:
                iter_per_epoch = int(self.trainer.num_training_batches)
            except Exception:
                iter_per_epoch = 0
        if iter_per_epoch <= 0:
            iter_per_epoch = 1
        iter_per_epoch = max(1, iter_per_epoch)
        self.lr_scheduler = LR_Scheduler(
            self.optimizer, 10, 1e-5, self.args.epochs, lr, 1e-6, iter_per_epoch
        )
        return self.optimizer
