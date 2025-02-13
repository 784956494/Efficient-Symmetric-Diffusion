from model import Network
import torch
import os

def load_model_optimizer(args, manifold, device):
    model_drift = Network(args.num_layers, args.dim , args.hidden_dim, args.dim, args.act, manifold, args.model_type, output_type='tangent')
    model_drift = model_drift.to(device)
    optimizer_drift = torch.optim.Adam(filter(lambda p: p.requires_grad, model_drift.parameters()), lr=args.lr,
                                    weight_decay=0.0)
    if manifold.name == 'Torus':
        model_diff = None
        optimizer_diff = None
    else:
        model_diff = Network(args.num_layers, args.dim , args.hidden_dim, args.dim, args.act, manifold, args.model_type, output_type='None')
        model_diff = model_diff.to(device)
        optimizer_diff = torch.optim.Adam(filter(lambda p: p.requires_grad, model_diff.parameters()), lr=args.lr,
                                        weight_decay=0.0)
    return model_drift, optimizer_drift, model_diff, optimizer_diff

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path