import torch
import torch.nn as nn
from torch import  nn
from src.model.utils.timestep_embed import TimestepEmbedding, Timesteps, TimestepEmbedderMDM
from src.model.utils.positional_encoding import PositionalEncoding
from src.model.utils.transf_utils import SkipTransformerEncoder, TransformerEncoderLayer
from src.model.utils.all_positional_encodings import build_position_encoding
from src.data.tools.tensors import lengths_to_mask
from src.model.utils.timestep_embed import TimestepEmbedderMDM
from src.model.DiT_models import DiTMotion


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

# architecure ablation
class DiT_Denoiser_CLS_Arch(nn.Module):

    def __init__(self,
                 nfeats: int = 263,
                 condition: str = "text",
                 motion_condition: str = None,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 text_encoded_dim: int = 768,
                 pred_delta_motion: bool = False,
                 use_sep: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.latent_dim = latent_dim
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.feat_comb_coeff = nn.Parameter(torch.tensor([1.0]))
        self.pose_proj_in_source = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj_in_target = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj_out = nn.Linear(self.latent_dim, nfeats)
        self.motion_condition = motion_condition

        # emb proj
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Linear(text_encoded_dim, self.latent_dim)
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        self.use_sep = True
        self.query_pos = PositionalEncoding(self.latent_dim, dropout = 0)
        self.cond_pos = PositionalEncoding(self.latent_dim, dropout = 0)# don't want to introduce noise here
        if self.motion_condition == "source":
            self.sep_token = nn.Parameter(torch.randn(1, self.latent_dim))

        # dit encoder
        self.dit_encoder = DiTMotion(
            in_channels=self.latent_dim,
            hidden_size=self.latent_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=ff_size / self.latent_dim,
        )
        # use torch transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.cond_encoder = nn.TransformerEncoder(encoder_layer,
                                                num_layers=kwargs.get('encoder_layers', 4))
        self.has_repr =  True
        self.source_head = build_mlp(self.latent_dim, 1024, kwargs['n_cls'])# classification
        self.target_head = build_mlp(self.latent_dim, 1024, kwargs['repr_dim'])
        self.use_target_mask = kwargs.get('use_target_mask', False) 
        self.ablation = kwargs.get('ablation', None)
        # assert self.ablation in ['raw_text', 'no_text', 'raw_source', 'raw_text_and_source'], 'ablation type not supported'
        

    def forward(self,
                noised_motion,
                timestep,
                in_motion_mask,
                text_embeds,
                condition_mask, 
                motion_embeds=None,
                lengths=None,
                src_len = None,
                tgt_len = None,
                **kwargs):
        if isinstance(src_len, list):
            src_len = torch.tensor(src_len)
            tgt_len = torch.tensor(tgt_len)
        # noised_motion: B, T, D
        # timestep: B
        # motion_embeds: T, B, D
        # len: B
        # proj_noised_motion

        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:
        motion_in_mask = in_motion_mask

        # time_embedding | text_embedding | frames_source | frames_target
        # 1 * lat_d | max_text * lat_d | max_frames * lat_d | max_frames * lat_d
        
        
        if self.condition in ["text", "text_uncond"]:
            # make it seq first
            text_embeds = text_embeds.permute(1, 0, 2)
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_embeds)
            else:
                text_emb_latent = text_embeds
                # source_motion_zeros = torch.zeros(*noised_motion.shape[:2], 
                #                             self.latent_dim, 
                #                             device=noised_motion.device)
                # aux_fake_mask = torch.zeros(condition_mask.shape[0], 
                #                             noised_motion.shape[0], 
                #                             device=noised_motion.device)
                # condition_mask = torch.cat((condition_mask, aux_fake_mask), 
                #                            1).bool().to(noised_motion.device)
            emb_latent_ori = text_emb_latent
            emb_latent = text_emb_latent
            # 1, B, D 

            if motion_embeds is not None:
                zeroes_mask = (motion_embeds == 0).all(dim=-1)
                if motion_embeds.shape[-1] != self.latent_dim:
                    motion_embeds_proj = self.pose_proj_in_source(motion_embeds)
                    motion_embeds_proj[zeroes_mask] = 0
                else:
                    motion_embeds_proj = motion_embeds
 
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        # 4. transformer
        # if self.diffusion_only:
        proj_noised_motion = self.pose_proj_in_target(noised_motion)
        
        # BUILD the mask now
        if motion_embeds is None:
            aug_mask = torch.cat((condition_mask[:, :text_emb_latent.shape[0]],
                                  motion_in_mask), 1)
        else:
            sep_token_mask = torch.ones((bs, self.sep_token.shape[0]),
                                        dtype=bool,
                                        device=noised_motion.device)
            aug_mask = torch.cat((
                            condition_mask[:, text_emb_latent.shape[0]:],
                            sep_token_mask,
                            motion_in_mask,
                            ), 1)

        # NOTE: condition encoding
        # emb_latent: T_Text, B, D
        # motion_embeds_proj: T_max, B, D
        motion_embeds_proj_ori = motion_embeds_proj
        cond_seq = torch.cat((emb_latent, motion_embeds_proj), dim=0)
        cond_seq = self.cond_pos(cond_seq)
        cond_seq_processed = self.cond_encoder(cond_seq, src_key_padding_mask=~condition_mask)
        T_Text = emb_latent.size(0)
        T_max = motion_embeds_proj.size(0)
        emb_latent_processed, motion_embeds_proj_processed = torch.split(cond_seq_processed, [T_Text, T_max], dim=0)
        emb_latent, motion_embeds_proj = emb_latent_processed, motion_embeds_proj_processed

        if motion_embeds is None:
            xseq = proj_noised_motion
        else:      
            sep_token_batch = torch.tile(self.sep_token, (bs,)).reshape(bs,
                                                                        -1)
            if self.ablation in ['raw_source', 'raw_text_and_source']:
                xseq = torch.cat((motion_embeds_proj_ori,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)
            else:
                xseq = torch.cat((motion_embeds_proj,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)

        xseq = self.query_pos(xseq)
        if self.use_target_mask:
            mask = None
        else:
            mask = aug_mask
            
        if self.ablation == 'no_text':
            condition = torch.zeros_like(emb_latent[0])
        elif self.ablation == 'raw_text' or self.ablation == 'raw_text_and_source':
            condition = emb_latent_ori[0]
        elif self.ablation == 'raw_source':
            condition = emb_latent[0]

        tokens = self.dit_encoder(xseq.permute(1, 0, 2), timestep, condition, mask=mask)
        # B, T, D
        tokens = tokens.permute(1, 0, 2)
        # if self.diffusion_only:
        if motion_embeds is not None:
            denoised_motion_proj = tokens
            if self.use_sep:
                useful_tokens = motion_embeds_proj.shape[0]+1
            else:
                useful_tokens = motion_embeds_proj.shape[0]
            denoised_motion_proj = denoised_motion_proj[useful_tokens:]
        else:
            denoised_motion_proj = tokens

        denoised_motion = self.pose_proj_out(denoised_motion_proj)
        denoised_motion[~motion_in_mask.T] = 0
        # zero for padded area
        # else:
        #     sample = tokens[:sample.shape[0]]
        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        denoised_motion = denoised_motion.permute(1, 0, 2)
        return denoised_motion

    def forward_with_repr(self,
                noised_motion,
                timestep,
                in_motion_mask,
                text_embeds,
                condition_mask, 
                motion_embeds=None,
                lengths=None,
                src_len = None,
                tgt_len = None,
                target_align_depth = 6, 
                return_attention = False, 
                **kwargs):
        if isinstance(src_len, list):
            src_len = torch.tensor(src_len)
            tgt_len = torch.tensor(tgt_len)
        # noised_motion: B, T, D
        # timestep: B
        # motion_embeds: T, B, D
        # len: B
        # proj_noised_motion

        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:
        motion_in_mask = in_motion_mask

        # time_embedding | text_embedding | frames_source | frames_target
        # 1 * lat_d | max_text * lat_d | max_frames * lat_d | max_frames * lat_d
        
        
        if self.condition in ["text", "text_uncond"]:
            # make it seq first
            text_embeds = text_embeds.permute(1, 0, 2)
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_embeds)
            else:
                text_emb_latent = text_embeds
                # source_motion_zeros = torch.zeros(*noised_motion.shape[:2], 
                #                             self.latent_dim, 
                #                             device=noised_motion.device)
                # aux_fake_mask = torch.zeros(condition_mask.shape[0], 
                #                             noised_motion.shape[0], 
                #                             device=noised_motion.device)
                # condition_mask = torch.cat((condition_mask, aux_fake_mask), 
                #                            1).bool().to(noised_motion.device)
            emb_latent_ori = text_emb_latent
            emb_latent = text_emb_latent
            # 1, B, D 

            if motion_embeds is not None:
                zeroes_mask = (motion_embeds == 0).all(dim=-1)
                if motion_embeds.shape[-1] != self.latent_dim:
                    motion_embeds_proj = self.pose_proj_in_source(motion_embeds)
                    motion_embeds_proj[zeroes_mask] = 0
                else:
                    motion_embeds_proj = motion_embeds
 
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        # 4. transformer
        # if self.diffusion_only:
        proj_noised_motion = self.pose_proj_in_target(noised_motion)
        
        # BUILD the mask now
        if motion_embeds is None:
            aug_mask = torch.cat((condition_mask[:, :text_emb_latent.shape[0]],
                                  motion_in_mask), 1)
        else:
            sep_token_mask = torch.ones((bs, self.sep_token.shape[0]),
                                        dtype=bool,
                                        device=noised_motion.device)
            aug_mask = torch.cat((
                            condition_mask[:, text_emb_latent.shape[0]:],
                            sep_token_mask,
                            motion_in_mask,
                            ), 1)
            # B, T_output

        # NOTE: condition encoding
        # emb_latent: T_Text, B, D
        # motion_embeds_proj: T_max, B, D
        motion_embeds_proj_ori = motion_embeds_proj
        cond_seq = torch.cat((emb_latent, motion_embeds_proj), dim=0)
        cond_seq = self.cond_pos(cond_seq)
        cond_seq_processed = self.cond_encoder(cond_seq, src_key_padding_mask=~condition_mask)
        T_Text = emb_latent.size(0)
        T_max = motion_embeds_proj.size(0)
        emb_latent_processed, motion_embeds_proj_processed = torch.split(cond_seq_processed, [T_Text, T_max], dim=0)
        emb_latent, motion_embeds_proj = emb_latent_processed, motion_embeds_proj_processed

        if motion_embeds is None:
            xseq = proj_noised_motion
        else:      
            sep_token_batch = torch.tile(self.sep_token, (bs,)).reshape(bs,
                                                                        -1)
            if self.ablation in ['raw_source', 'raw_text_and_source']:
                xseq = torch.cat((motion_embeds_proj_ori,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)
            else:
                xseq = torch.cat((motion_embeds_proj,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)

        xseq = self.query_pos(xseq)
        if self.use_target_mask:
            mask = None
        else:
            mask = aug_mask
        if self.ablation == 'no_text':
            condition = torch.zeros_like(emb_latent[0])
        elif self.ablation == 'raw_text' or self.ablation == 'raw_text_and_source':
            condition = emb_latent_ori[0]
        elif self.ablation == 'raw_source':
            condition = emb_latent[0]

        if return_attention:
            tokens, target_repr, attention_mask = self.dit_encoder.forward_with_repr_att(xseq.permute(1, 0, 2), \
            timestep, condition, depth=target_align_depth, mask=mask)
        else:
            tokens, target_repr = self.dit_encoder.forward_with_repr(xseq.permute(1, 0, 2), \
            timestep, condition, depth=target_align_depth, mask=mask)
        # B, T, D
        tokens = tokens.permute(1, 0, 2)
        # if self.diffusion_only:
        if motion_embeds is not None:
            denoised_motion_proj = tokens
            if self.use_sep:
                useful_tokens = motion_embeds_proj.shape[0]+1
            else:
                useful_tokens = motion_embeds_proj.shape[0]
            denoised_motion_proj = denoised_motion_proj[useful_tokens:]
        else:
            denoised_motion_proj = tokens

        denoised_motion = self.pose_proj_out(denoised_motion_proj)
        denoised_motion[~motion_in_mask.T] = 0
        # zero for padded area
        # else:
        #     sample = tokens[:sample.shape[0]]
        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        denoised_motion = denoised_motion.permute(1, 0, 2)

        target_repr = self.target_head(target_repr[:, T_max+1:]) # B, T_target, D
        source_repr = self.source_head(motion_embeds_proj) # B, T_source, D
        if return_attention:
            return denoised_motion, source_repr.permute(1,0,2), target_repr, attention_mask
        return denoised_motion, source_repr.permute(1,0,2), target_repr

    def forward_with_guidance(self,
                              noised_motion,
                              timestep,
                              in_motion_mask,
                              text_embeds,
                              condition_mask,
                              guidance_motion,
                              guidance_text_n_motion, 
                              motion_embeds=None,
                              lengths=None,
                              inpaint_dict=None,
                              max_steps=None,
                              prob_way='3way',
                              **kwargs):
        # if motion embeds is None
        # TODO put here that you have tow
        # implement 2 cases for that case
        # text unconditional more or less 2 replicas
        # timestep
        if max_steps is not None:
            curr_ts = timestep[0].item()
            g_m = max(1, guidance_motion*2*curr_ts/max_steps)
            guidance_motion = g_m
            g_t_tm = max(1, guidance_text_n_motion*2*curr_ts/max_steps)
            guidance_text_n_motion = g_t_tm

        if motion_embeds is None:
            half = noised_motion[: len(noised_motion) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, timestep,
                                    in_motion_mask=in_motion_mask,
                                    text_embeds=text_embeds,
                                    condition_mask=condition_mask, 
                                    motion_embeds=motion_embeds,
                                    lengths=lengths)
            uncond_eps, cond_eps_text = torch.split(model_out, len(model_out) // 2,
                                                     dim=0)
            # make it BxSxfeatures
            if inpaint_dict is not None:
                import torch.nn.functional as F
                source_mot = inpaint_dict['start_motion'].permute(1, 0, 2)
                if source_mot.shape[1] >= uncond_eps.shape[1]:
                    source_mot = source_mot[:, :uncond_eps.shape[1]]
                else:
                    pad = uncond_eps.shape[1] - source_mot.shape[1]
                    # Pad the tensor on the second dimension (time)
                    source_mot = F.pad(source_mot, (0, 0, 0, pad), 'constant', 0)

                mot_len = source_mot.shape[1]
                # concat mask for all the frames
                mask_src_parts = inpaint_dict['mask'].unsqueeze(1).repeat(1,
                                                                      mot_len,
                                                                      1)
                uncond_eps = uncond_eps*(mask_src_parts) + source_mot*(~mask_src_parts)
                cond_eps_text = cond_eps_text*(mask_src_parts) + source_mot*(~mask_src_parts)
            half_eps = uncond_eps + guidance_text_n_motion * (cond_eps_text - uncond_eps) 
            eps = torch.cat([half_eps, half_eps], dim=0)
        else:
            third = noised_motion[: len(noised_motion) // 3]
            combined = torch.cat([third, third, third], dim=0)
            model_out = self.forward(combined, timestep,
                                     in_motion_mask=in_motion_mask,
                                     text_embeds=text_embeds,
                                     condition_mask=condition_mask, 
                                     motion_embeds=motion_embeds,
                                     lengths=lengths)
            # For exact reproducibility reasons, we apply classifier-free guidance on only
            # three channels by default. The standard approach to cfg applies it to all channels.
            # This can be done by uncommenting the following line and commenting-out the line following that.
            # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
            # eps, rest = model_out[:, :3], model_out[:, 3:]
            uncond_eps, cond_eps_motion, cond_eps_text_n_motion = torch.split(model_out,
                                                                            len(model_out) // 3,
                                                                            dim=0)
            if inpaint_dict is not None:
                import torch.nn.functional as F
                source_mot = inpaint_dict['start_motion'].permute(1, 0, 2)
                if source_mot.shape[1] >= uncond_eps.shape[1]:
                    source_mot = source_mot[:, :uncond_eps.shape[1]]
                else:
                    pad = uncond_eps.shape[1] - source_mot.shape[1]
                    # Pad the tensor on the second dimension (time)
                    source_mot = F.pad(source_mot, (0, 0, 0, pad), 'constant', 0)

                mot_len = source_mot.shape[1]
                # concat mask for all the frames
                mask_src_parts = inpaint_dict['mask'].unsqueeze(1).repeat(1,
                                                                      mot_len,
                                                                      1)
                uncond_eps = uncond_eps*(~mask_src_parts) + source_mot*mask_src_parts
                cond_eps_text = cond_eps_text*(~mask_src_parts) + source_mot*mask_src_parts
                cond_eps_text_n_motion = cond_eps_text_n_motion*(~mask_src_parts) + source_mot*mask_src_parts
            if prob_way=='3way':
                third_eps = uncond_eps + guidance_motion * (cond_eps_motion - uncond_eps) + \
                            guidance_text_n_motion * (cond_eps_text_n_motion - cond_eps_motion)
            if prob_way=='2way':
                third_eps = uncond_eps + guidance_text_n_motion * (cond_eps_text_n_motion - uncond_eps)

            eps = torch.cat([third_eps, third_eps, third_eps], dim=0)
        return eps

import torch

def test_TMED_DiTMotionDenoiser():
    # 设置随机种子以保证测试的可复现性
    torch.manual_seed(42)

    # 测试参数
    batch_size = 8
    seq_length = 100  # 序列长度
    nfeats = 263
    latent_dim = 256
    text_encoded_dim = 768

    # 创建测试输入数据
    noised_motion = torch.randn(batch_size, seq_length, nfeats)  # 输入的噪声运动数据 (N, T, nfeats)
    timestep = torch.randint(0, 1000, (batch_size,))  # 随机生成时间步 (N,)
    text_embeds = torch.randn(batch_size, 1, text_encoded_dim)  # 随机生成文本嵌入 (N, D)
    in_motion_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)  # 假设无填充
    condition_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)  # 条件掩码
    motion_embeds = torch.randn(seq_length, batch_size, nfeats)  # 随机生成运动嵌入 (N, T, latent_dim)

    # 创建模型实例
    model = DiT_Denoiser(
        nfeats=nfeats,
        latent_dim=latent_dim,
        text_encoded_dim=text_encoded_dim,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation='gelu',
        motion_condition = 'source'
    )

    # 将模型设置为评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        output = model(noised_motion, timestep, in_motion_mask, text_embeds, condition_mask, motion_embeds=motion_embeds)

    # 检查输出的形状
    expected_shape = (batch_size, seq_length, nfeats)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    # 检查输出是否包含合理值 (例如无NaN)
    assert not torch.isnan(output).any(), "Output contains NaN values, which is unexpected."

    print("Test passed: TMED_DiTMotionDenoiser handles 1D inputs correctly!")

# 运行测试
if __name__ == "__main__":
    test_TMED_DiTMotionDenoiser()
