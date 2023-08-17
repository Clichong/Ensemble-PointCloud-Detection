import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union
from timm.models.layers import Mlp, DropPath
from timm.layers import PatchDropout
from tools.ensemble_utils.timm_model.module import Block, SwimBlock

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
    def __init__(self, max_c, alpha=0.5):
        super(WeightMergeNet, self).__init__()
        # alpha -> ∞, the same with nms
        # self.alpha = 10
        # w = (1.0 / torch.linspace(1, max_c, max_c)) ** self.alpha       # func: (1 / x)^α
        # self.w = nn.Parameter(w, requires_grad=True)

        # weight merge with confidence
        self.alpha = alpha
        # self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.w = nn.Parameter(torch.ones(max_c), requires_grad=True)   # weight: [k]

    def forward(self, x, conf=None):
        """
        Input
            x:          (g, k, 7)
            confidence: (g, k, 1)
        Return:
            Tensor:     (g, 7)
        """
        if conf is None:
            w = self.w

            out = x * w[:, None]
            k = x.any(-1).sum(-1)

            # norm by the weights sum
            w_ = x.new([self.w[:k_].sum() for k_ in k])     # keep the same device and dtype
            res = (out * self.w[None, :, None]).sum(1) / w_[:, None]
            return res

        if conf is not None:
            assert x.shape[:2] == conf.shape[:2]

            # weight add with conf
            w = self.w
            out_c = x * conf
            out_w = x * w[None, :, None]

            # norm add by weight
            k = x.any(-1).sum(-1)
            w_ = x.new([w[:k_].sum() for k_ in k])      # (g, )
            out_w = out_w.sum(1) / w_[:, None]

            # norm add by confidence
            conf_ = x.new([conf[i][:k_].sum() for i, k_ in enumerate(k)])   # (g, )
            out_c = out_c.sum(1) / conf_[:, None]

            res = self.alpha * out_w + (1 - self.alpha) * out_c
            return res


class AttentionMergeNet(nn.Module):
    def __init__(
            self,
            max_c,
            global_pool: str = 'token',
            embed_dim: int = 7,             # box size: (g, k, 7)
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

        self.reg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))    # (1, 1, 7)
        embed_len = max_c + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)  # (K+1, 7)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
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
            x = torch.cat((self.reg_token.expand(x.shape[0], -1, -1), x), dim=1)  # (g, 1, d) + (g, k, d) -> (g, k+1, d)
        x = x + self.pos_embed  # (g, k+1, d) + (1, k+1, d)
        return self.pos_drop(x)

    def forward(self, x, conf=None):
        """
        Input
            x:          (g, k, 7)
        Return:
            Tensor:     (g, 7)
        """
        base_ = x[:, 0]
        x = self._pos_embed(x)
        # x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        if self.global_pool == 'avg':
            x = x[self.num_prefix_tokens:].mean(dim=0)
            return x
        else:
            reg_token = x[:, 0]
            return reg_token + base_


class SwimMergeNet(nn.Module):
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
            reg_token: bool = True,
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
            block_fn: Callable = SwimBlock,
            mlp_layer: Callable = Mlp,
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        # assert reg_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if reg_token else 0

        self.reg_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if reg_token else None      # (1, 1, 7)
        k_dim = max_c + self.num_prefix_tokens
        # embed_len = max_c + self.num_prefix_tokens
        # self.pos_embed = nn.Parameter(torch.randn(embed_len, embed_dim) * .02)  # (K+1, 7)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                max_c=k_dim,
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

    def forward(self, x, conf=None):
        """
        Input
            x: (g, k, 7)
        Return:
            Tensor: (g, 7)
        """
        base_ = x[:, 0]
        if self.reg_token is not None:
            x = torch.cat((self.reg_token.expand(x.shape[0], -1, -1), x), dim=1)  # (g, 1, d) + (g, k, d) -> (g, k+1, d)
        x = self.blocks(x)
        x = self.norm(x)
        if self.global_pool == 'avg':
            x = x[self.num_prefix_tokens:].mean(dim=0)
            return x
        else:
            reg_token = x[:, 0]
            return reg_token + base_


if __name__ == '__main__':
    k = 5
    g = 100
    # x = torch.zeros([10, 7])
    # x[:k, :] = torch.rand(k, 7)

    x = torch.rand([2, k, 7])
    conf = torch.rand([2, k, 1])
    model = WeightMergeNet(max_c=k)
    # model = AttentionMergeNet(max_c=k, depth=3)
    # model = SwimMergeNet(max_c=k, depth=3, init_values=True, reg_token=True)
    print(x)

    y = model(x, conf)
    print(y)
