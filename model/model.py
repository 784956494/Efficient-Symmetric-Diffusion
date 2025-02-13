import torch
from utils.math_utils import get_activation
from .resnet import ResNet

class MLP(torch.nn.Module):
    def __init__(self, num_layers, in_dim, hid_dim, out_dim, act, bias=True):
        super().__init__()
        hidden_shapes = [hid_dim] * num_layers

        self.linears = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        for n in range(len(hidden_shapes)+1):
            if n==0:
                self.linears.append(torch.nn.Linear(in_dim, hidden_shapes[0], bias=bias))
            elif n==len(hidden_shapes):
                self.linears.append(torch.nn.Linear(hidden_shapes[-1], out_dim, bias=bias))
            else:
                self.linears.append(torch.nn.Linear(hidden_shapes[n-1], hidden_shapes[n], bias=bias))
            
        if act is not None:
            for n in range(len(hidden_shapes)):
                if n  < len(hidden_shapes)-1:
                    self.acts.append(get_activation(act, in_dim=hidden_shapes[n], out_dim=hidden_shapes[n+1]))
                else:
                    self.acts.append(get_activation(act, in_dim=hidden_shapes[n], out_dim=out_dim))
        else:
            self.acts = None
            
    def forward(self, x):
        for _, linear in enumerate(self.linears):
            x = linear(x)
            if _ < len(self.linears)-1 and self.acts is not None:
                x = self.acts[_](x)
        return x
    
class Network(torch.nn.Module):
    def __init__(self, num_layers, in_dim, hid_dim, out_dim, act, manifold, type='MLP', output_type='tangent'):
        super().__init__()
        if manifold.name == 'Torus':
            self.in_dim=in_dim
            self.out_dim=out_dim
        elif manifold.name == 'Special Orthogonal':
            if output_type=='tangent': # add dimension for torus
                self.in_dim = in_dim * (in_dim + 1)
                self.out_dim = out_dim * (out_dim + 1)
            else:
                self.in_dim = in_dim * (in_dim)
                self.out_dim = out_dim * (out_dim)
            # self.torus_model = MLP(4, in_dim + 1, hid_dim, out_dim, act)
        elif manifold.name == 'Unitary':
            self.in_dim = 2 * in_dim ** 2
            self.out_dim = 2 * out_dim ** 2
            # self.torus_model = MLP(4, in_dim + 1, hid_dim, out_dim, act)
        else:
            raise NotImplementedError('Only Torus, SO(n), U(n) are supported..')
        
        if type=='MLP':
            self.layer = MLP(num_layers, self.in_dim + 1, hid_dim, self.out_dim, act)
        elif type=='ResNet':
            self.layer = ResNet(self.in_dim, self.out_dim, hid_dim, 128, num_layers)
        else:
            raise NotImplementedError('Only MLP or ResNet supported for the neural network')
        
        self.manifold = manifold
        self.feat_dim = in_dim
        self.network_type = type
        self.output_type=output_type
    
    def shape(self, x):
        if self.manifold.name == 'Unitary':
            x = torch.view_as_real(x)
            x = x.reshape(-1, 2 * self.feat_dim ** 2)
            return x
        elif len(x.shape) == 3:
            x = x.reshape(-1, self.feat_dim * (self.feat_dim))
        return x
    
    def unshape(self, x):
        if self.manifold.name == 'Unitary':
            y = x.reshape(-1, self.feat_dim, self.feat_dim, 2).contiguous()
            y = torch.view_as_complex(y)
            return y
        x = x.reshape(-1, self.feat_dim, self.feat_dim)
        return x

    def forward(self, x, t, x_torus=None):
        y = self.shape(x)
        if x_torus is not None:
            y = torch.cat([y, x_torus], dim=-1)
        if self.network_type == 'MLP':
            # concat time and input
            output = self.layer(torch.cat([y, t], dim=-1))
        else:
            # time embedding instead
            output = self.layer(y, t)
        if self.manifold.name == 'Unitary' or self.manifold.name == 'Special Orthogonal':
            if x_torus is not None:
                phase = output[..., -self.feat_dim:]
                output = output[..., :-self.feat_dim]
                output = self.unshape(output)
                # project to tangent for the drift term
                output = self.manifold.proju(x, output)  
                return output, phase
            else:
                output = self.unshape(output)
                if self.output_type == 'tangent':
                    output = self.manifold.proju(x, output)
                return output
        return output