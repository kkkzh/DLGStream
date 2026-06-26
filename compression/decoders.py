import math
import torch
import scipy
import numpy as np
import torch.nn as nn
from torch import Tensor
# from fast_pytorch_kmeans import KMeans
from torch.nn import Module, Parameter, init
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple, _ntuple
# import torchac
import os, contextlib
suppress_output = True # 可以设置成 False 来调试
if suppress_output:
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                import torchac
            except ImportError as e:
                # 可以选择重新抛出异常
                raise e
            except Exception as e:
                # 捕获其他可能的异常
                raise e
else:
    import torchac

epsilon = 1e-6
def get_dft_matrix(conv_dim, channels):
    dft = torch.zeros(conv_dim,channels)
    for i in range(conv_dim):
        for j in range(channels):
            # Each row of dft is a bias vector
            dft[i,j] = math.cos(torch.pi/channels*(i+0.5)*j)/math.sqrt(channels) 
            dft[i,j] = dft[i,j]*(math.sqrt(2) if j>0 else 1)
    return dft


class StraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class DecoderLayer(Module):

    def __init__(self, in_features: int, out_features: int, ldecode_matrix: str, bias: bool = False) -> None:
        super(DecoderLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if 'dft' in ldecode_matrix:
            self.dft = Parameter(get_dft_matrix(in_features, out_features), requires_grad=False)
        if 'dft' in ldecode_matrix:
            self.scale = Parameter(torch.empty((1,out_features)))
        else:
            self.scale = Parameter(torch.empty((in_features,out_features)))
        if bias:
            self.shift = Parameter(torch.empty(1,out_features))
        else:
            self.register_parameter('shift', None)

        self.ldecode_matrix = ldecode_matrix
        if ldecode_matrix == 'dft_fixed':
            self.scale.requires_grad_(False)
            if not bias:
                self.shift.requires_grad_(False)

    def reset_parameters(self, param=1.0, init_type = 'normal') -> None:
        if init_type == 'normal':
            init.normal_(self.scale, std=param)
        elif init_type == 'uniform':
            init.uniform_(self.scale, -param, param)
        elif init_type == 'constant':
            init.constant_(self.scale, val=param)
        elif init_type == 'zero':
            init.zeros_(self.scale)
        if self.shift is not None:
            init.zeros_(self.shift)

    def clamp(self, val: float = 0.5) -> None:
        with torch.no_grad():
            self.scale.clamp_(-val, val)

    def forward(self, input: Tensor) -> Tensor:
        if 'dft' in self.ldecode_matrix:
            w_out = torch.matmul(input,self.dft)*self.scale+(self.shift if self.shift is not None else 0)
        else:
            w_out = torch.matmul(input,self.scale)+(self.shift if self.shift is not None else 0)
        return w_out

    def invert(self, output: Tensor) -> Tensor:
        shift = self.shift if self.shift is not None else 0
        if self.in_features == 1 and self.out_features == 1:
            input = (output-shift)/(self.scale+epsilon)
        else:
            if 'dft' in self.ldecode_matrix:
                input = torch.linalg.lstsq(self.dft.T,((output-shift)/self.scale).T).solution.T
            else:
                input = torch.linalg.lstsq(self.scale.T,(output-shift).T).solution.T
        return input
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.shift is not None
        )

class LatentDecoder(Module):

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        ldecode_matrix: str,
        use_shift: bool = True,
        norm: str = 'none',
        num_layers_dec: int = 0,
        hidden_dim_dec: int = 0,
        activation: str = 'relu',
        final_activation: str = 'none',
        clamp_weights: float = 0.0,
        ldec_std: float = 1.0,
        use_gumbel: bool = False,
        diff_sampling: bool = False,
        **kwargs,
    ) -> None:
        super(LatentDecoder, self).__init__()
        latent_dim = feature_dim if latent_dim == 0 else latent_dim
        self.ldecode_matrix = ldecode_matrix
        self.channels = feature_dim
        self.latent_dim = latent_dim
        self.div = nn.Parameter(torch.ones(latent_dim),requires_grad=False)
        self.norm = norm
        # self.div = 1.0
        self.num_layers_dec =  num_layers_dec
        if num_layers_dec > 0 and hidden_dim_dec == 0:
            hidden_dim_dec = feature_dim
            self.hidden_dim_dec = _ntuple(num_layers_dec)(hidden_dim_dec)
        self.use_shift = use_shift
        act_dict = {
                    'none':torch.nn.Identity(), 'sigmoid':torch.nn.Sigmoid(), 'tanh':torch.nn.Tanh(),
                    'relu':torch.nn.ReLU(),
                    }
        self.act = act_dict[activation]
        self.final_activation = act_dict[final_activation]
        self.clamp_weights = clamp_weights
        
        layers = []
        for l in range(num_layers_dec):
            feature_dim = self.hidden_dim_dec[l]
            feature_dim = latent_dim if feature_dim == 0 else feature_dim
            layers.append(DecoderLayer(latent_dim, feature_dim, ldecode_matrix, bias=self.use_shift))
            layers.append(self.act)
            latent_dim = feature_dim
        feature_dim = self.channels
        layers.append(DecoderLayer(latent_dim,feature_dim,ldecode_matrix,bias=self.use_shift))

        # self.use_gumbel = nn.Parameter(torch.Tensor([use_gumbel]).bool(),requires_grad=False)
        # self.temperature = nn.Parameter(torch.Tensor([1.0]),requires_grad=False)
        self.use_gumbel = use_gumbel
        self.temperature = 1.0
        self.layers = nn.Sequential(*layers)
        # self.reset_parameters('normal', ldec_std)
        self.reset_parameters('zero', ldec_std)
        self.diff_sampling = diff_sampling
        
    def normalize(self, input:Tensor):
        if self.norm == "min_max":
            self.div.data = torch.max(torch.abs(input),dim=0)[0]
        elif self.norm == "mean_std":
            self.div.data = torch.std(input,dim=0)
        self.div.data = torch.max(self.div,torch.ones_like(self.div))
        
    def reset_parameters(self, init_type, param=0.5) -> None:
        for layer in list(self.layers.children()):
            if isinstance(layer, DecoderLayer):
                layer.reset_parameters(param,init_type)

    def get_scale(self):
        assert self.num_layers_dec == 0, "Can only get scale for 0 hidden layers decoder!"
        return list(self.layers.children())[0].scale

    def get_shift(self):
        assert self.num_layers_dec == 0, "Can only get scale for 0 hidden layers decoder!"
        return list(self.layers.children())[0].shift
    
    def clamp(self, val: float = 0.2) -> None:
        for layer in list(self.layers.children()):
            if isinstance(layer, DecoderLayer):
                layer.clamp(val)

    def size(self, use_torchac=False):
        return sum([p.numel()*torch.finfo(p.dtype).bits for p in self.parameters()])

    def scale_norm(self):
        return list(self.layers.children())[0].scale.norm()

    def scale_grad_norm(self):
        return list(self.layers.children())[0].scale.grad.norm()
    
    def forward(self, weight: Tensor) -> Tensor:
        if self.use_gumbel:
            weightf = torch.floor(weight) if self.diff_sampling else StraightThrough.apply(weight)
            weightc = weightf+1
            logits_f = -torch.tanh(torch.clamp(weight-weightf, min=-1+epsilon, max=1-epsilon)).unsqueeze(-1)/self.temperature
            logits_c = -torch.tanh(torch.clamp(weightc-weight, min=-1+epsilon, max=1-epsilon)).unsqueeze(-1)/self.temperature
            logits = torch.cat((logits_f,logits_c),dim=-1)
            dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(self.temperature, logits=logits)
            sample = dist.rsample() if self.diff_sampling else dist.sample()
            weight = weightf*sample[...,0]+weightc*sample[...,1]
        else:
            weight = StraightThrough.apply(weight)
        w_out = self.layers(weight/self.div)
        w_out = self.final_activation(w_out)
        if self.clamp_weights>0.0:
            w_out = torch.clamp(w_out, min=-self.clamp_weights, max=self.clamp_weights)
        return w_out
    
    def invert(self, output: Tensor) -> Tensor:
        with torch.no_grad():
            x = output
            prev_layer = None
            for idx,layers in enumerate(list(self.layers.children())[::-1]):
                if isinstance(layers, DecoderLayer):
                    x = layers.invert(x)
                elif isinstance(layers, torch.nn.Identity):
                    continue 
                elif isinstance(layers, torch.nn.ReLU):
                    if isinstance(prev_layer, DecoderLayer):
                        min_x = x.min(dim=0)[0]
                        shift_x = torch.min(min_x,torch.zeros_like(min_x)).unsqueeze(0)
                        if prev_layer.shift is not None:
                            prev_layer.shift.data -= torch.matmul(shift_x,prev_layer.scale)
                        else:
                            prev_layer.shift = Parameter(-torch.matmul(shift_x,prev_layer.scale),requires_grad=False)
                            prev_layer.shift.device = prev_layer.scale.device
                        x -= shift_x
                elif isinstance(layers, torch.nn.Sigmoid):
                    if isinstance(prev_layer, DecoderLayer):
                        max_x, min_x = x.max(dim=0)[0], x.min(dim=0)[0]
                        diff_x = max_x-min_x
                        diff_x = torch.max(diff_x,torch.ones_like(diff_x))
                        prev_layer.scale.data /= diff_x.unsqueeze(-1)
                        x /= diff_x.unsqueeze(0)

                        min_x = x.min(dim=0)[0]
                        shift_x = torch.min(min_x,torch.zeros_like(min_x)).unsqueeze(0)
                        if prev_layer.shift is not None:
                            prev_layer.shift.data -= torch.matmul(shift_x,prev_layer.scale)
                        else:
                            prev_layer.shift = Parameter(-torch.matmul(shift_x,prev_layer.scale),requires_grad=False)
                            prev_layer.shift.device = prev_layer.scale.device
                        x -= shift_x

                    x = torch.clamp(x, min=epsilon, max=1-epsilon)
                    x = torch.log(x/(1-x))
                elif isinstance(layers, torch.nn.Tanh):
                    if isinstance(prev_layer, DecoderLayer):
                        max_x, min_x = x.max(dim=0)[0], x.min(dim=0)[0]
                        diff_x = max_x-min_x
                        diff_x = torch.max(diff_x,torch.ones_like(diff_x)*2)
                        prev_layer.scale.data /= diff_x.unsqueeze(-1)
                        x /= diff_x.unsqueeze(0)

                        min_x = x.min(dim=0)[0]
                        shift_x = torch.min(min_x+1,torch.zeros_like(min_x)).unsqueeze(0)
                        if prev_layer.shift is not None:
                            prev_layer.shift.data -= torch.matmul(shift_x,prev_layer.scale)
                        else:
                            prev_layer.shift = Parameter(-torch.matmul(shift_x,prev_layer.scale),requires_grad=False)
                            prev_layer.shift.device = prev_layer.scale.device
                        x -= shift_x

                    x = torch.clamp(x, min=-1+epsilon, max=1-epsilon)
                    x = torch.atanh(x)
                prev_layer = layers
            return x*self.div
        
    def infer(self, weight: Tensor) -> Tensor:
        weight = StraightThrough.apply(weight)
        weight
        w_out = self.layers(weight/self.div)
        w_out = self.final_activation(w_out)
        if self.clamp_weights>0.0:
            w_out = torch.clamp(w_out, min=-self.clamp_weights, max=self.clamp_weights)
        return w_out


# class CodebookQuantize(torch.nn.Module):
#     def __init__(self,
#                  codebook_bitwidth: int,
#                  codebook_dim: int,
#                  use_gumbel: bool = False):
#         super().__init__()
#         self.codebook_bitwidth = codebook_bitwidth
#         self.codebook_size = 2**codebook_bitwidth
#         self.codebook_dim = codebook_dim
#         self.codebook = nn.Parameter(torch.empty((self.codebook_size, self.codebook_dim)))
#         self.temperature = 1.0
#         self.use_gumbel = use_gumbel
#         self.reset_parameters('constant', 0.0)
#
#     def reset_parameters(self, init_type, param=1.0) -> None:
#         if init_type == 'normal':
#             init.normal_(self.codebook, std=param)
#         elif init_type == 'uniform':
#             init.uniform_(self.codebook, -param, param)
#         elif init_type == 'constant':
#             init.constant_(self.codebook, val=param)
#
#     def init(self, output: Tensor):
#         with torch.no_grad():
#             kmeans = KMeans(n_clusters=self.codebook_size, mode='euclidean', init_method='++', max_iter=500)
#             kmeans.fit_predict(output)
#             self.codebook.data = kmeans.centroids.to(self.codebook)
#
#     def forward(self, weights):
#         if not self.use_gumbel:
#             indices = torch.argmax(weights, dim=-1)
#             quantized_weights = self.codebook[indices]
#         else:
#             softmax_weights = torch.nn.functional.gumbel_softmax(weights, tau=self.temperature)
#             quantized_weights = torch.matmul(softmax_weights, self.codebook)
#         return quantized_weights
#
#     def invert(self, output: Tensor) -> Tensor:
#         with torch.no_grad():
#             if not self.use_gumbel:
#                 distances = torch.cdist(output, self.codebook)
#                 indices = torch.argmin(distances, dim=-1)
#                 input = torch.nn.functional.one_hot(indices, num_classes=self.codebook_size).to(output)
#             else:
#                 softmax_out = torch.matmul(torch.linalg.pinv(self.codebook.T),output.T).T
#                 # softmax_out = torch.lstsq(self.codebook.T, output.T).solution.T
#                 # softmax_out = torch.Tensor(scipy.optimize.nnls(
#                 #                                                 self.codebook.T.detach().cpu().numpy(),
#                 #                                                 output.T.detach().cpu().numpy()
#                 #                                                 )
#                 #                             ).to(output).T
#                 input = torch.log(torch.clamp(softmax_out, min=epsilon))*self.temperature
#
#         return input
#
#     def infer(self, weights):
#         indices = torch.argmax(weights, dim=-1)
#         quantized_weights = self.codebook[indices]
#         return quantized_weights
#
#     def size(self, use_torchac=False):
#         return self.codebook.numel()*torch.finfo(self.codebook.dtype).bits


# Class for identity decoder with placeholder variables/functions
class DecoderIdentity(Module):

    def __init__(
        self,
    ) -> None:
        super(DecoderIdentity, self).__init__()
        self.latent_dim = 1
        self.num_layers_dec = 0
        self.shift = False
        self.norm = 'none'
        self.use_gumbel = False
        self.temperature = 1.0
        self.div = 1.0
        
    # For compatibility with Decoder
    def reset_parameters(self, init_type, param=1.0) -> None:
        return

    def forward(self, input: Tensor) -> Tensor:
        # print(input.min(),input.max())
        return input

    def scale_norm(self):
        return 1
    
    def scale_grad_norm(self):
        return 1
    
    def size(self, use_torchac=False) -> int:
        return 0
    
    def invert(self, output: Tensor) -> Tensor:
        return output


class CompressedLatents(object):
    def compress(self, latent):
        assert latent.dim() == 2, "Latent should be 2D"
        self.num_latents, self.latent_dim = latent.shape
        flattened = latent.flatten()
        flattened = flattened * 1

        weight = torch.round(flattened).int()
        unique_vals, counts = torch.unique(weight, return_counts=True)
        probs = counts / torch.sum(counts)
        tail_idx = torch.where(probs <= 1.0e-4)[0]
        tail_vals = unique_vals[tail_idx]
        self.tail_locs = {}
        for val in tail_vals:
            self.tail_locs[val.item()] = torch.where(weight == val)[0].detach().cpu()
            weight[weight == val] = unique_vals[counts.argmax()]
        unique_vals, counts = torch.unique(weight, return_counts=True)
        probs = counts / torch.sum(counts)
        weight = weight.detach().cpu()

        cdf = torch.cumsum(probs, dim=0)
        cdf = torch.cat((torch.Tensor([0.0]).to(cdf), cdf))
        cdf = cdf / cdf[-1:]  # Normalize the final cdf value just to keep torchac happy
        cdf = cdf.unsqueeze(0).repeat(flattened.size(0), 1)

        mapping = {val.item(): idx.item() for val, idx in zip(unique_vals, torch.arange(unique_vals.shape[0]))}
        self.mapping = mapping
        weight.apply_(mapping.get)
        byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), weight.to(torch.int16))

        self.byte_stream, self.mapping, self.cdf = byte_stream, mapping, cdf[0].detach().cpu().numpy()

    def uncompress(self):
        import torchac
        cdf = torch.tensor(self.cdf).unsqueeze(0).repeat(self.num_latents * self.latent_dim, 1)
        weight = torchac.decode_float_cdf(cdf, self.byte_stream)
        weight = weight.to(torch.float32)

        inverse_mapping = {v: k for k, v in self.mapping.items()}
        weight.apply_(inverse_mapping.get)
        for val, locs in self.tail_locs.items():
            weight[locs] = val
        weight = weight.view(self.num_latents, self.latent_dim)
        weight /= 1
        return weight