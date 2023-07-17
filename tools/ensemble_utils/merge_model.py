import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union
from timm.models.layers import Mlp, DropPath
from tools.ensemble_utils.timm_model.module import Block, ResPostBlock

class MergeMLP(nn.Module):
    def __init__(self, max_c, hidden_c=16):
        super(MergeMLP, self).__init__()
        self.operate = nn.Sequential(
            # Rearrange('k d -> d k'),
            nn.Linear(max_c, hidden_c),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_c, hidden_c),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_c, 1)
        )

    def forward(self, x):
        """
            Whatever x is (max_c,d) return a box (1,d)
            这里的输出需要是直接的预测框结果
        """
        x = self.operate(x.permute([1, 0]))
        return x.permute([1, 0])


class WeightMergeNet(nn.Module):
    def __init__(self, max_c):
        super(WeightMergeNet, self).__init__()
        w = (1.0 / torch.linspace(1, max_c, max_c)) ** 2       # func: (1 / x)^2
        self.w = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        """
        Input
            x: (max_c, 7)
        Return:
            Tensor: (7)
        """
        out = x * self.w[:, None]
        k = x.any(-1).sum()
        # norm by the weights sum
        return out[:k, :].sum(dim=0) / self.w[:k].sum()


class AttentionMergeNet(nn.Module):
    def __init__(
            self,
            max_c,
            global_pool: str = 'token',
            embed_dim: int = 7,             # box size: (k, 7)
            depth: int = 12,
            num_heads: int = 1,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            # embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        # self.no_embed_class = no_embed_class
        # self.grad_checkpointing = False

        self.reg_token = nn.Parameter(torch.zeros(1, embed_dim))    # (1, 7)
        embed_len = max_c + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(embed_len, embed_dim) * .02)  # (K+1, 7)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        # if patch_drop_rate > 0:
        #     self.patch_drop = PatchDropout(
        #         patch_drop_rate,
        #         num_prefix_tokens=self.num_prefix_tokens,
        #     )
        # else:
        #     self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

    def _pos_embed(self, x):
        if self.reg_token is not None:
            x = torch.cat([self.reg_token, x], dim=0)    # (1, 7) + (max_c, 7)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x):
        """
        Params:
            x: (max, 7)
        Return:
            out: Tensor: (7, )
        """
        base_ = x[0]
        x = self._pos_embed(x)
        # x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.global_pool:
            x = x[self.num_prefix_tokens:].mean(dim=0) if self.global_pool == 'avg' else x[0]
        return x + base_


if __name__ == '__main__':
    k = 5
    x = torch.zeros([10, 7])
    x[:k, :] = torch.rand(k, 7)
    model = AttentionMergeNet(max_c=10)
    y = model(x)
    print(y)
