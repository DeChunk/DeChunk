"""
Model factory + training wrapper for the UNIFIED single-forward VLA v2
(asymmetric chunk lengths; see models/vla_model_fm_unified_2.py).

MetaWorld port of DINOV3-HDF5-Myself-3/models/model_runner_unified_2.py:
the only change is load_norm_stats() accepting the MetaWorld stats file
(top-level key 'metaworld') in addition to 'robotwin2'.
"""
import torch
import torch.nn as nn
import logging
import json
from transformers import AutoModel, AutoConfig

from .vla_model_fm_unified_2 import VLAModel, calc_flow_matching_loss_unified

logger = logging.getLogger(__name__)


class ModelFactory:
    @staticmethod
    def create_vision_encoder(checkpoint_path, dtype=torch.bfloat16, device="cuda"):
        """
        load & freeze DINOv3 ViT.

        Returns:
            model: DINOv3 model (eval mode, frozen)
            hidden_size: int, model hidden dimension
            num_register_tokens: int, register token num
            patch_size: int, ViT patch size
        """
        logger.info(f"Loading frozen DINOv3 vision encoder from {checkpoint_path}...")

        config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)

        model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)

        # Freeze
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        hidden_size = getattr(config, "hidden_size", None)
        num_register_tokens = getattr(config, "num_register_tokens", 4)
        patch_size = getattr(config, "patch_size", 16)

        if hidden_size is None:
            # fallback: dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
                out = model(pixel_values=dummy)
                hidden_size = out.last_hidden_state.shape[-1]

        logger.info(
            f"DINOv3 loaded: hidden_size={hidden_size}, "
            f"num_register_tokens={num_register_tokens}, patch_size={patch_size}"
        )
        return model, hidden_size, num_register_tokens, patch_size

    @staticmethod
    def create_action_model(config, dino_hidden_size, num_dino_layers, task_cond_dim=None):
        """create unified VLAModel v2 (asymmetric chunks)"""
        logger.info("Initializing unified VLAModel v2...")

        model_cfg = config.model
        ae_cfg = model_cfg.action_expert
        ve_cfg = model_cfg.vision_encoder

        dino_feat_dims = tuple([dino_hidden_size] * num_dino_layers)

        # ---- multi layer feature fusion mode ----
        fusion_mode = ve_cfg.get('fusion_mode', 'per_layer')
        concat_cfg = ve_cfg.get('concat', {})
        concat_proj_type = concat_cfg.get('proj_type', 'linear') if concat_cfg else 'linear'
        concat_pre_norm = concat_cfg.get('pre_norm', True) if concat_cfg else True
        concat_out_dim = concat_cfg.get('out_dim', None) if concat_cfg else None
        if concat_out_dim is None:
            concat_out_dim = dino_hidden_size

        # ---- asymmetric chunk lengths ----
        action_len = config.common.action_chunk_size
        next_action_len = config.common.get('next_chunk_size', action_len)

        # ---- next-chunk design switches ----
        next_attend_current = model_cfg.get('next_attend_current', False)
        separate_next_output_proj = model_cfg.get('separate_next_output_proj', True)
        next_pos_mode = model_cfg.get('next_pos_mode', 'continuous')

        logger.info(f"Fusion mode: {fusion_mode} "
                    f"(concat_out_dim={concat_out_dim}, proj={concat_proj_type}, pre_norm={concat_pre_norm})")
        if task_cond_dim is not None:
            logger.info(f"Task condition ENABLED (dim={task_cond_dim})")
        logger.info(f"Chunk lengths: current={action_len}, next={next_action_len} | "
                    f"next_attend_current={next_attend_current}, "
                    f"separate_next_output_proj={separate_next_output_proj}, "
                    f"next_pos_mode={next_pos_mode}")

        model = VLAModel(
            action_dim=config.common.action_dim,
            proprio_dim=config.common.state_dim,
            hidden_dim=ae_cfg.hidden_size,
            action_len=action_len,
            next_action_len=next_action_len,
            proprio_len=config.common.proprio_len,
            depth=ae_cfg.depth,
            num_heads=ae_cfg.num_heads,
            dino_feat_dims=dino_feat_dims,
            vlm_num_queries=ae_cfg.vlm_adapter_num_queries,
            adapter_depth=ae_cfg.adapter_depth,
            num_registers=model_cfg.num_registers,
            state_inject_start=0,
            task_cond_dim=task_cond_dim,
            fusion_mode=fusion_mode,
            concat_proj_type=concat_proj_type,
            concat_pre_norm=concat_pre_norm,
            concat_out_dim=concat_out_dim,
            next_attend_current=next_attend_current,
            separate_next_output_proj=separate_next_output_proj,
            next_pos_mode=next_pos_mode,
        )
        return model


class VLAWrapper(nn.Module):
    def __init__(self,
                 vision_encoder,
                 action_model,
                 time_sampler,
                 feat_layers,
                 include_cls_register,
                 num_register_tokens,
                 device,
                 dtype,
                 norm_stats_path,
                 train_config=None,
                 ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.action_model = action_model
        self.time_sampler = time_sampler
        self.feat_layers = list(feat_layers)
        self.include_cls_register = include_cls_register
        self.num_register_tokens = num_register_tokens

        self.device = device
        self.dtype = dtype

        # param only for training
        self.train_config = train_config
        if self.train_config is not None:
            self.time_mu = train_config['time_mu']
            self.time_sigma = train_config['time_sigma']
            self.use_next_chunk_pred = train_config['use_next_chunk_pred']
            self.lambda_next_chunk = train_config['lambda_next_chunk']
            self.independent_next_time = train_config['independent_next_time']
        else:
            # default param for infer
            self.time_mu = 0.0
            self.time_sigma = 1.0
            self.use_next_chunk_pred = False
            self.lambda_next_chunk = 0.0
            self.independent_next_time = True

        logger.info(f"VLAWrapper (unified v2) initialized. feat_layers={self.feat_layers}, "
                    f"include_cls_register={self.include_cls_register}")
        if self.use_next_chunk_pred:
            logger.info(f"Next-Chunk Prediction ENABLED (unified single forward, "
                        f"lambda={self.lambda_next_chunk}, "
                        f"independent_next_time={self.independent_next_time})")

        self.load_norm_stats(norm_stats_path)

    def load_norm_stats(self, path):
        logger.info(f"Loading normalization stats from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)

        # Prefer 'robotwin2' for back-compat; otherwise use 'metaworld', else the
        # first top-level key. The MetaWorld stats file uses key 'metaworld'.
        if 'robotwin2' in data:
            stats = data['robotwin2']
        elif 'metaworld' in data:
            stats = data['metaworld']
        else:
            first_key = next(iter(data.keys()))
            logger.warning(f"Unknown stats key set {list(data.keys())}; using '{first_key}'")
            stats = data[first_key]

        action_stats = stats['action']
        state_stats = stats['state']

        act_min = torch.tensor(action_stats['min'], dtype=torch.float32)
        act_max = torch.tensor(action_stats['max'], dtype=torch.float32)
        self.register_buffer('action_min', act_min)
        self.register_buffer('action_max', act_max)
        logger.info(f"Loaded Action stats - Dim: {len(action_stats['min'])}")

        state_min = torch.tensor(state_stats['min'], dtype=torch.float32)
        state_max = torch.tensor(state_stats['max'], dtype=torch.float32)
        self.register_buffer('state_min', state_min)
        self.register_buffer('state_max', state_max)
        logger.info(f"Loaded State stats - Dim: {len(state_stats['min'])}")

    @torch.no_grad()
    def get_vision_features(self, pixel_values):
        """
        get DINOv3 hidden states.

        Args:
            pixel_values: (B, 3, H, W)

        Returns:
            List[Tensor(B, N, hidden_size)] length = len(feat_layers)
            if include_cls_register=True, N = 1 + num_register_tokens + num_patches
            else N = num_patches (the 1 + num_register_tokens tokens are dropped)
        """
        pixel_values = pixel_values.to(self.device, self.dtype)
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states

        feats_list = []
        for layer_idx in self.feat_layers:
            h = hidden_states[layer_idx]   # (B, 1+R+P, D)
            if not self.include_cls_register:
                skip = 1 + self.num_register_tokens
                h = h[:, skip:, :]
            feats_list.append(h)
        return feats_list

    def _normalize_tensor(self, x, min_val, max_val):
        min_v = min_val.to(device=x.device, dtype=x.dtype)
        max_v = max_val.to(device=x.device, dtype=x.dtype)

        denominator = max_v - min_v
        denominator[denominator < 1e-6] = 1.0

        norm_x = 2 * (x - min_v) / denominator - 1
        return norm_x

    def normalize_action(self, action):
        return self._normalize_tensor(action, self.action_min, self.action_max)

    def normalize_state(self, state):
        return self._normalize_tensor(state, self.state_min, self.state_max)

    def denormalize_action(self, norm_action):
        action_min = self.action_min.to(device=norm_action.device, dtype=norm_action.dtype)
        action_max = self.action_max.to(device=norm_action.device, dtype=norm_action.dtype)

        denominator = action_max - action_min
        denominator[denominator < 1e-6] = 1.0

        action = (norm_action + 1) / 2 * denominator + action_min
        return action

    def forward(self, batch):
        # 1. Vision feature
        pixel_values = batch['pixel_values']      # (B, 3, H, W)
        dino_features_list = self.get_vision_features(pixel_values)

        # 2. Action / State
        x1_raw = batch['action_sequence'].to(self.device, self.dtype)
        qpos_raw = batch['state'].to(self.device, self.dtype)

        if qpos_raw.dim() == 2:
            qpos_raw = qpos_raw.unsqueeze(1)

        # 3. Normalize
        x1 = self.normalize_action(x1_raw)
        qpos = self.normalize_state(qpos_raw)
        qpos_history = qpos

        # 4. Next-Chunk GT
        x1_next = None
        if self.use_next_chunk_pred and batch.get('next_action_sequence') is not None:
            x1_next_raw = batch['next_action_sequence'].to(self.device, self.dtype)
            x1_next = self.normalize_action(x1_next_raw)

        # 4b. Task Condition
        task_cond = None
        if batch.get('task_cond') is not None:
            task_cond = batch['task_cond'].to(self.device, self.dtype)

        # 5. Unified Flow Matching Loss (single forward)
        loss, info_dic = calc_flow_matching_loss_unified(
            self.action_model,
            x1=x1,
            dino_features_list=dino_features_list,
            qpos_history=qpos_history,
            task_cond=task_cond,
            time_sampler=self.time_sampler,
            time_mu=self.time_mu,
            time_sigma=self.time_sigma,
            x1_next=x1_next,
            use_next_chunk_pred=self.use_next_chunk_pred,
            lambda_next_chunk=self.lambda_next_chunk,
            independent_next_time=self.independent_next_time,
        )

        return loss, info_dic
