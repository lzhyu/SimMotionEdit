import torch
import torch.nn as nn
from torch import  nn
from src.model.utils.timestep_embed import TimestepEmbedding, Timesteps, TimestepEmbedderMDM
from src.model.utils.positional_encoding import PositionalEncoding
from src.model.utils.transf_utils import SkipTransformerEncoder, TransformerEncoderLayer
from src.model.utils.all_positional_encodings import build_position_encoding
from src.data.tools.tensors import lengths_to_mask
from src.model.utils.timestep_embed import TimestepEmbedderMDM

class TMED_denoiser(nn.Module):

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

        print("num_heads", num_heads)
        self.latent_dim = latent_dim
        self.pred_delta_motion = pred_delta_motion
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.feat_comb_coeff = nn.Parameter(torch.tensor([1.0]))
        self.pose_proj_in_source = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj_in_target = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj_out = nn.Linear(self.latent_dim, nfeats)
        self.first_pose_proj = nn.Linear(self.latent_dim, nfeats)
        self.motion_condition = motion_condition

        # emb proj
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time from text_encoded_dim to latent_dim
            self.embed_timestep = TimestepEmbedderMDM(self.latent_dim)

            # FIXME me TODO this            
            # self.time_embedding = TimestepEmbedderMDM(self.latent_dim)
            
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Linear(text_encoded_dim, self.latent_dim)
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        self.use_sep = use_sep
        self.query_pos = PositionalEncoding(self.latent_dim, dropout)
        self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        if self.motion_condition == "source":
            if self.use_sep:
                self.sep_token = nn.Parameter(torch.randn(1, self.latent_dim))

        # use torch transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                                num_layers=num_layers)

    def forward(self,
                noised_motion,
                timestep,
                in_motion_mask,
                text_embeds,
                condition_mask, 
                motion_embeds=None,
                lengths=None,
                **kwargs):
        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:
        motion_in_mask = in_motion_mask

        # time_embedding | text_embedding | frames_source | frames_target
        # 1 * lat_d | max_text * lat_d | max_frames * lat_d | max_frames * lat_d
        
        
        # 1. time_embeddingno
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(noised_motion.shape[1]).clone()
        time_emb = self.embed_timestep(timesteps).to(dtype=noised_motion.dtype)
        # make it S first
        # time_emb = self.time_embedding(time_emb).unsqueeze(0)
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
            emb_latent = torch.cat((time_emb, text_emb_latent), 0)

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

        if motion_embeds is None:
            xseq = torch.cat((emb_latent, proj_noised_motion), axis=0)
        else:
            if self.use_sep:

                sep_token_batch = torch.tile(self.sep_token, (bs,)).reshape(bs,
                                                                         -1)
                xseq = torch.cat((emb_latent, motion_embeds_proj,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)
            else:
                xseq = torch.cat((emb_latent, motion_embeds_proj,
                                  proj_noised_motion), axis=0)
        # if self.ablation_skip_connection:
        #     xseq = self.query_pos(xseq)
        #     tokens = self.encoder(xseq)
        # else:
        #     # adding the timestep embed
        #     # [seqlen+1, bs, d]
        #     # todo change to query_pos_decoder
        xseq = self.query_pos(xseq)
        # BUILD the mask now
        if motion_embeds is None:
            time_token_mask = torch.ones((bs, time_emb.shape[0]),
                                        dtype=bool, device=xseq.device)
            aug_mask = torch.cat((time_token_mask,
                                  condition_mask[:, :text_emb_latent.shape[0]],
                                  motion_in_mask), 1)
        else:
            time_token_mask = torch.ones((bs, time_emb.shape[0]),
                                        dtype=bool,
                                        device=xseq.device)
            if self.use_sep:
                sep_token_mask = torch.ones((bs, self.sep_token.shape[0]),
                                        dtype=bool,
                                        device=xseq.device)
            if self.use_sep:
                aug_mask = torch.cat((time_token_mask,
                                condition_mask[:, :text_emb_latent.shape[0]],
                                condition_mask[:, text_emb_latent.shape[0]:],
                                sep_token_mask,
                                motion_in_mask,
                                ), 1)
            else:
                aug_mask = torch.cat((time_token_mask,
                                condition_mask[:, :text_emb_latent.shape[0]],
                                condition_mask[:, text_emb_latent.shape[0]:],
                                motion_in_mask,
                                ), 1)
        tokens = self.encoder(xseq, src_key_padding_mask=~aug_mask)

        # if self.diffusion_only:
        if motion_embeds is not None:
            denoised_motion_proj = tokens[emb_latent.shape[0]:]
            if self.use_sep:
                useful_tokens = motion_embeds_proj.shape[0]+1
            else:
                useful_tokens = motion_embeds_proj.shape[0]
            denoised_motion_proj = denoised_motion_proj[useful_tokens:]
        else:
            denoised_motion_proj = tokens[emb_latent.shape[0]:]

        denoised_motion = self.pose_proj_out(denoised_motion_proj)
        if self.pred_delta_motion and motion_embeds is not None:
            import torch.nn.functional as F
            tgt_size = len(denoised_motion)
            if len(denoised_motion) > len(motion_embeds):
                pad_for_src = tgt_size - len(motion_embeds)
                motion_embeds = F.pad(motion_embeds, 
                                      (0, 0, 0, 0, 0, pad_for_src))
            denoised_motion = denoised_motion + motion_embeds[:tgt_size]

        denoised_motion[~motion_in_mask.T] = 0
        # zero for padded area
        # else:
        #     sample = tokens[:sample.shape[0]]
        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        denoised_motion = denoised_motion.permute(1, 0, 2)
        return denoised_motion

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

from ptflops import get_model_complexity_info
import time
from tqdm import tqdm
if __name__ == "__main__":
    batch_size = 1
    seq_length = 120  # 序列长度
    nfeats = 263
    latent_dim = 768
    text_encoded_dim = 768
    model = TMED_denoiser(
        nfeats=nfeats,
        latent_dim=latent_dim,
        text_encoded_dim=text_encoded_dim,
        ff_size = 2048,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        activation='gelu',
        motion_condition = 'source',
    )
    def inp(dummy):
        noised_motion = torch.randn(batch_size, seq_length, nfeats)  # 输入的噪声运动数据 (N, T, nfeats)
        timestep = torch.randint(0, 1000, (batch_size,))  # 随机生成时间步 (N,)
        text_embeds = torch.randn(batch_size, 1, text_encoded_dim)  # 随机生成文本嵌入 (N, D)
        in_motion_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)  # 假设无填充
        condition_mask = torch.ones(batch_size, seq_length + 1, dtype=torch.bool)  # 条件掩码
        motion_embeds = torch.randn(seq_length, batch_size, nfeats)  # 随机生成运动嵌入 (N, T, latent_dim)
        return dict(noised_motion=noised_motion, timestep = timestep, \
                in_motion_mask=in_motion_mask, text_embeds=text_embeds, condition_mask=condition_mask, \
                    motion_embeds=motion_embeds)

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
                                           print_per_layer_stat=True, verbose=True, input_constructor=inp)
    print(macs)
    print(params)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    start = time.time()
    for _ in tqdm(range(100)):
        model(**inp(None))
    end = time.time()
    avg = (end-start)/100
    print(avg)
