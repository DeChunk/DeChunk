"""
Unified single-forward VLA v2 (asymmetric chunk lengths).

Changes vs vla_model_fm_unified.py:
  - Current / next chunk lengths are independent (e.g. 16 current + 48 next).
  - `next_attend_current` switch: whether next-chunk tokens may attend to the
    current action tokens (default False — blocks the shortcut). Base tokens
    NEVER attend to next-chunk tokens regardless of this switch, so dropping
    the next tokens at inference still leaves base outputs unchanged.
  - Next-chunk tokens get their own type embedding (`type_emb_action_next`)
    and a positional embedding controlled by `next_pos_mode`:
    "continuous" (default) encodes the honest absolute positions A..A+A'-1 —
    all distinct from the current chunk's 0..A-1; "restart" uses 0..A'-1,
    whose sincos prefix coincides with the current chunk's (ablation only).
  - `separate_next_output_proj` switch: use a dedicated output projection for
    the next chunk (default True) or share `output_proj`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Standard Transformer SinCos positional encoding.
    Returns: (1, length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    pos = torch.arange(length, dtype=torch.float32)
    grid = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (grid / (embed_dim // 2)))

    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.unsqueeze(0)


# --- Time Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.to(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modulate(x, shift, scale):
    """Per-token adaLN modulation. x/shift/scale: (B, S, D)."""
    return x * (1 + scale) + shift


class VisualFeatureAdapter(nn.Module):
    """Perceiver-style adapter: learnable queries cross-attend to vision tokens, output (B, num_queries, hidden_dim)."""
    def __init__(self, feat_dim, hidden_dim, num_queries=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        self.input_norm = nn.LayerNorm(feat_dim)
        self.feature_proj = nn.Linear(feat_dim, hidden_dim)

        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, features):
        """
        features: (B, N, feat_dim)
        Returns: (B, num_queries, hidden_dim)
        """
        B = features.shape[0]
        memory = self.input_norm(features)
        memory = self.feature_proj(memory)
        tgt = self.query_embed.expand(B, -1, -1)
        out = self.transformer_decoder(tgt, memory)
        return out


class MultiLayerConcatFusion(nn.Module):
    """
    multi layer DINO feature fusion (concat mode):
      - LayerNorm
      - concat in channel dim → (B, N, L*feat_dim)
      - Linear / MLP  → (B, N, out_dim)
    """
    def __init__(self, feat_dim, num_layers, out_dim, proj_type="linear", pre_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        if pre_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(num_layers)])

        in_dim = feat_dim * num_layers
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        elif proj_type == "linear":
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported concat proj_type: {proj_type}")

    def forward(self, feats_list):
        """feats_list: List[(B, N, feat_dim)]"""
        assert len(feats_list) == self.num_layers, \
            f"MultiLayerConcatFusion expects {self.num_layers} layers, got {len(feats_list)}"
        if self.pre_norm:
            feats_list = [ln(f) for ln, f in zip(self.layer_norms, feats_list)]
        x = torch.cat(feats_list, dim=-1)   # (B, N, L*feat_dim)
        return self.proj(x)                 # (B, N, out_dim)


class DiTBlock(nn.Module):
    """
    DiT block with Self-Attention + MLP, conditioned via adaLN-zero on a
    PER-TOKEN time embedding (B, S, D). Base-sequence tokens are modulated by
    the current-chunk time t; next-chunk action tokens by t_next.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # 6 params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb, attn_mask=None):
        """x: (B, S, D); emb: (B, S, D) per-token time embedding."""
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
            self.adaLN_modulation(emb).chunk(6, dim=-1)

        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn1(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False
        )[0]

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm)
        return x


class VLAModel(nn.Module):
    """
    Unified single-forward VLA v2 (asymmetric chunks).

    Sequence layout (training):
        [current_actions(A), proprio(P), registers(R), (task_cond)(1), obs(Q...), next_actions(A')]
    with A = action_len and A' = next_action_len possibly different.
    Inference: the trailing next_actions block is simply never appended.
    """
    def __init__(self,
                 action_dim=14,
                 proprio_dim=16,
                 hidden_dim=512,
                 num_heads=4,
                 depth=12,
                 action_len=16,
                 next_action_len=48,
                 proprio_len=1,
                 num_registers=2,
                 dino_feat_dims=(1024,),     # feat_dim, len = feat layers
                 vlm_num_queries=64,
                 adapter_depth=2,
                 state_inject_start=0,
                 task_cond_dim=None,
                 # --- multi layer feature fusion mode ---
                 fusion_mode="per_layer",
                 concat_proj_type="linear",
                 concat_pre_norm=True,
                 concat_out_dim=None,
                 # --- next-chunk design switches ---
                 next_attend_current=False,
                 separate_next_output_proj=True,
                 next_pos_mode="continuous",     # "continuous" | "restart"
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_len = action_len
        self.next_action_len = next_action_len
        self.proprio_len = proprio_len
        self.num_registers = num_registers
        self.state_inject_start = state_inject_start
        self.num_dino_layers = len(dino_feat_dims)
        self.use_task_cond = task_cond_dim is not None
        self.next_attend_current = next_attend_current
        self.separate_next_output_proj = separate_next_output_proj
        assert next_pos_mode in ("continuous", "restart"), \
            f"Unknown next_pos_mode: {next_pos_mode}"
        self.next_pos_mode = next_pos_mode

        self.fusion_mode = fusion_mode
        assert fusion_mode in ("per_layer", "concat"), f"Unknown fusion_mode: {fusion_mode}"

        # --- 1. Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # --- 2. Projections & PosEmb ---
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)

        # Next-chunk positional embedding. NOTE on sincos: values depend only
        # on the absolute position index, so a "restart at 0" sincos would make
        # the first min(A, A') positions EXACTLY EQUAL to the current chunk's —
        # the strongest possible position-wise correspondence channel.
        # "continuous" (default): next positions = action_len .. action_len+A'-1
        #   → every next-chunk position vector is distinct from every current
        #   one, and the encoding is the honest absolute temporal position.
        # "restart": independent 0..A'-1 buffer (prefix coincides with the
        #   current chunk's by sincos construction; kept for ablation).
        self.register_buffer('action_pos_emb', get_1d_sincos_pos_embed(hidden_dim, action_len))
        if next_pos_mode == "continuous":
            full = get_1d_sincos_pos_embed(hidden_dim, action_len + next_action_len)
            self.register_buffer('next_action_pos_emb', full[:, action_len:, :])
        else:
            self.register_buffer('next_action_pos_emb',
                                 get_1d_sincos_pos_embed(hidden_dim, next_action_len))
        self.register_buffer('proprio_pos_emb', get_1d_sincos_pos_embed(hidden_dim, proprio_len))

        # Type embeddings: action (current), action_next, proprio + DINO
        self.type_emb_action = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_action_next = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_proprio = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        num_dino_type_emb = 1 if fusion_mode == "concat" else self.num_dino_layers
        self.type_emb_dino_layers = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in range(num_dino_type_emb)
        ])

        nn.init.normal_(self.type_emb_action, std=0.02)
        nn.init.normal_(self.type_emb_action_next, std=0.02)
        nn.init.normal_(self.type_emb_proprio, std=0.02)
        for emb in self.type_emb_dino_layers:
            nn.init.normal_(emb, std=0.02)

        # --- 3. Register Tokens ---
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- 3b. Task Condition Token (appended to the DiT input sequence) ---
        if self.use_task_cond:
            self.task_cond_proj = nn.Sequential(
                nn.LayerNorm(task_cond_dim),
                nn.Linear(task_cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.type_emb_task_cond = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.type_emb_task_cond, std=0.02)

        # --- 4. DINO adapters ---
        if fusion_mode == "concat":
            # one adapter
            fusion_out_dim = concat_out_dim if concat_out_dim is not None else dino_feat_dims[0]
            self.concat_fusion = MultiLayerConcatFusion(
                feat_dim=dino_feat_dims[0],
                num_layers=self.num_dino_layers,
                out_dim=fusion_out_dim,
                proj_type=concat_proj_type,
                pre_norm=concat_pre_norm,
            )
            self.dino_adapters = nn.ModuleList([
                VisualFeatureAdapter(
                    feat_dim=fusion_out_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                    num_heads=num_heads, num_layers=adapter_depth, dropout=0.
                )
            ])
        else:
            # per_layer: per adapter (NOT shared across layers)
            self.dino_adapters = nn.ModuleList([
                VisualFeatureAdapter(
                    feat_dim=feat_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                    num_heads=num_heads, num_layers=adapter_depth, dropout=0.
                )
                for feat_dim in dino_feat_dims
            ])

        # --- 5. Core Transformer Blocks ---
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])

        # --- 6. Output Head(s): shared trunk, per-role output projections ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        if self.separate_next_output_proj:
            self.output_proj_next = nn.Linear(hidden_dim, action_dim)

    def _build_unified_mask(self, num_base, num_next, device):
        """
        Block-causal attention mask for the unified sequence, bool (S, S) with
        True == "not allowed to attend" (PyTorch MultiheadAttention convention).

        Token layout: [base tokens (current actions + proprio + registers +
        (task_cond) + obs), next-chunk action tokens].

        Rules:
          - No base token may attend to next-chunk tokens (always), so dropping
            them at inference changes nothing.
          - Next-chunk tokens attend to current action tokens only when
            self.next_attend_current is True (default False: the current→next
            shortcut stays blocked).
        """
        S = num_base + num_next
        mask = torch.zeros(S, S, dtype=torch.bool, device=device)
        mask[:num_base, num_base:] = True                       # base  -/-> next (always)
        if not self.next_attend_current:
            mask[num_base:, :self.action_len] = True            # next  -/-> current actions
        return mask

    def forward(self,
                t,
                noisy_actions,
                qpos_history=None,
                dino_features_list=None,    # List[Tensor(B, N_l, feat_dim_l)]
                task_cond=None,             # (B, task_cond_dim)
                t_next=None,                # (B,) next-chunk denoising time
                noisy_actions_next=None,    # (B, next_action_len, action_dim) noised next chunk
                ):
        """
        Training (unified): pass t_next + noisy_actions_next → the next-chunk
        action tokens are appended and processed in the same forward pass.
        Inference: leave them None → the sequence contains only base tokens and
        produces exactly the same current-chunk outputs as the masked training
        forward (attention is the only cross-token op, and base tokens never
        attend to next-chunk tokens).
        """
        B = noisy_actions.shape[0]

        # 1. Action / Proprio tokens
        x_action = self.action_proj(noisy_actions) + \
                   self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                   self.type_emb_action

        x_proprio_real = self.proprio_proj(qpos_history) + \
                         self.proprio_pos_emb[:, :qpos_history.shape[1], :] + \
                         self.type_emb_proprio

        # State (injected at block `state_inject_start`, zero-init before that)
        x_proprio = torch.zeros_like(x_proprio_real)

        tokens_list = [x_action, x_proprio]
        proprio_start = x_action.shape[1]
        proprio_end = proprio_start + x_proprio.shape[1]

        # 2. Register Tokens
        if self.num_registers > 0:
            regs = self.register_tokens.expand(B, -1, -1)
            tokens_list.append(regs)

        # 2b. Task Condition Token
        if self.use_task_cond:
            assert task_cond is not None, "use_task_cond=True, forward without task_cond"
            tc_proj = self.task_cond_proj(task_cond).unsqueeze(1)     # (B, 1, hidden_dim)
            tokens_list.append(tc_proj + self.type_emb_task_cond)

        # 3. Multi-layer DINO Feature Tokens
        if dino_features_list is not None:
            assert len(dino_features_list) == self.num_dino_layers, \
                f"Expected {self.num_dino_layers} DINO feature layers, got {len(dino_features_list)}"
            if self.fusion_mode == "concat":
                fused = self.concat_fusion(dino_features_list)       # (B, N, fusion_out_dim)
                cond = self.dino_adapters[0](fused)                  # (B, Q, hidden_dim)
                tokens_list.append(cond + self.type_emb_dino_layers[0])
            else:
                for layer_idx, feats in enumerate(dino_features_list):
                    cond = self.dino_adapters[layer_idx](feats)
                    tokens_list.append(cond + self.type_emb_dino_layers[layer_idx])

        x_base = torch.cat(tokens_list, dim=1)
        num_base = x_base.shape[1]

        # 4. Per-token time embedding: base tokens use t, next-chunk tokens use t_next
        t_emb = self.time_mlp(t)                     # (B, D)
        emb_base = t_emb.unsqueeze(1).expand(B, num_base, self.hidden_dim)

        use_next = noisy_actions_next is not None
        if use_next:
            assert t_next is not None, "noisy_actions_next given but t_next is None"
            t_emb_next = self.time_mlp(t_next)       # (B, D)
            x_action_next = self.action_proj(noisy_actions_next) + \
                            self.next_action_pos_emb[:, :noisy_actions_next.shape[1], :] + \
                            self.type_emb_action_next
            num_next = x_action_next.shape[1]

            x = torch.cat([x_base, x_action_next], dim=1)
            emb = torch.cat([
                emb_base,
                t_emb_next.unsqueeze(1).expand(B, num_next, self.hidden_dim),
            ], dim=1)
            attn_mask = self._build_unified_mask(num_base, num_next, x.device)
        else:
            x = x_base
            emb = emb_base
            attn_mask = None

        # 5. Transformer Blocks
        state_injected = False
        for i, block in enumerate(self.blocks):
            if i == self.state_inject_start and not state_injected:
                x[:, proprio_start:proprio_end, :] = x_proprio_real
                state_injected = True
            x = block(x, emb, attn_mask=attn_mask)

        # 6. Output Head(s) (shared norm; per-role projections)
        x = self.final_norm(x)
        out = {"final_pred": self.output_proj(x[:, :self.action_len, :])}
        if use_next:
            proj_next = self.output_proj_next if self.separate_next_output_proj else self.output_proj
            out["final_pred_next"] = proj_next(x[:, num_base:, :])
        return out


def _sample_time(bs, device, time_sampler, time_mu, time_sigma):
    if time_sampler == "uniform":
        return torch.rand(bs, device=device)
    elif time_sampler == "logit_normal":
        normal_samples = torch.randn(bs, device=device)
        normal_samples = normal_samples * time_sigma + time_mu
        return torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unsupported time_sampler: {time_sampler}")


def calc_flow_matching_loss_unified(
    model,
    x1,
    dino_features_list,
    qpos_history,
    task_cond=None,
    time_sampler="uniform",
    time_mu=0.0,
    time_sigma=1.0,
    # Next-Chunk Prediction (unified single forward, asymmetric lengths)
    x1_next=None,
    use_next_chunk_pred=False,
    lambda_next_chunk=0.5,
    independent_next_time=True,
    # Optional externally-sampled noise/time. SOAR post-training passes its own
    # (x0, t0) so the base loss and the off-trajectory rollout share them.
    x0=None,
    t0=None,
):
    """
    Unified Flow Matching Loss (v2):
      - one forward pass predicts both the current chunk (length A, time t) and
        the next chunk (length A', possibly different, time t_next);
      - loss = mse_current + lambda_next_chunk * mse_next.
    """
    device = x1.device
    bs = x1.shape[0]

    # 1. Current chunk: noise, time, interpolation, target velocity
    if x0 is None:
        x0 = torch.randn_like(x1)
    t = _sample_time(bs, device, time_sampler, time_mu, time_sigma) if t0 is None else t0
    t_expand = t.view(bs, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1
    target_v = x1 - x0

    # 2. Next chunk: independent noise; time independent or shared (config switch)
    x_t_next, t_next, target_v_next = None, None, None
    if use_next_chunk_pred and x1_next is not None:
        x0_next = torch.randn_like(x1_next)
        if independent_next_time:
            t_next = _sample_time(bs, device, time_sampler, time_mu, time_sigma)
        else:
            t_next = t
        t_next_expand = t_next.view(bs, 1, 1)
        x_t_next = (1 - t_next_expand) * x0_next + t_next_expand * x1_next
        target_v_next = x1_next - x0_next

    # 3. Single unified forward
    preds = model(t,
                  noisy_actions=x_t,
                  dino_features_list=dino_features_list,
                  task_cond=task_cond,
                  qpos_history=qpos_history,
                  t_next=t_next,
                  noisy_actions_next=x_t_next)

    # 4. Losses
    loss_mse = F.mse_loss(preds["final_pred"], target_v)

    loss_next_chunk = torch.tensor(0.0, device=device)
    if "final_pred_next" in preds:
        loss_next_chunk = F.mse_loss(preds["final_pred_next"], target_v_next)

    loss = loss_mse + lambda_next_chunk * loss_next_chunk

    return loss, {
        "pred_v": preds["final_pred"],
        "target_v": target_v,
        "loss_mse": loss_mse.item(),
        "loss_next_chunk": loss_next_chunk.item(),
    }
