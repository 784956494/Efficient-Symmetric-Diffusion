import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
import math
import random
import torch.nn as nn
import argparse
from utils.initialize import load_model_optimizer, create_dir
from tqdm import tqdm
from model import Sampler, Beta_Scheduler
from utils.plot_utils import plot_samples
from manifold import Torus, SOn, Un
from utils.eval_utils import loglikelihood, C2ST

def load_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed 

class Trainer(object):
    def __init__(self, args, manifold, train_loader, target=None, beta_scheduler=None):
        super(Trainer, self).__init__()
        self.train_loader = train_loader
        self.args = args
        self.device = args.device
        self.manifold = manifold
        self.beta_scheduler = beta_scheduler
        self.model_drift, self.opt_drift, self.model_diff, self.opt_diff = load_model_optimizer(self.args, manifold, self.device)
        self.sampler = Sampler(manifold, beta_scheduler, self.model_drift, self.model_diff, device=args.device)
        self.target = target
    def train(self):
        t_start = time.time()
        for epoch in tqdm(range(self.args.epoch), total=self.args.epoch, desc="Training Model"):
            self.train_drift = []
            self.model_drift.train()

            if self.model_diff:
                self.model_diff.train()
                self.train_diff = []

            for _, train_b in enumerate(self.train_loader):
                x = train_b[0].to(self.device)
                loss_drift, loss_diff = self.manifold.loss(x, self.model_drift, self.model_diff, self.beta_scheduler)
                self.opt_drift.zero_grad()
                loss_drift.backward()

                if loss_diff:
                    self.opt_diff.zero_grad()
                    loss_diff.backward()

                self.opt_drift.step()
                self.train_drift.append(loss_drift.item())
                if loss_diff:
                    self.opt_diff.step() 
                    self.train_diff.append(loss_diff.item())

            if (epoch + 1) % 30 == 0 or epoch == 0:
                mean_train_drift = np.mean(self.train_drift)
                if self.model_diff:
                    mean_train_diff = np.mean(self.train_diff)
                    print((f'{epoch+1:03d} | {time.time()-t_start:.4f}s | train drift: {mean_train_drift:.4f} | train diff: {mean_train_diff:.4f} |'))
                else:
                    print((f'{epoch+1:03d} | {time.time()-t_start:.4f}s | train drift: {mean_train_drift:.4f} | Not Training Diffusion Term'))

            if (epoch + 1) % 50 == 0 or epoch == 0:
                if self.args.ckpt_directory:
                    create_dir(self.args.ckpt_directory)
                    checkpoint_path_drift = '{}drift_model_{}.pt'.format(self.args.ckpt_directory, epoch + 1)
                    torch.save(self.model_drift.state_dict(), checkpoint_path_drift)
                    if self.model_diff:
                        checkpoint_path_diff = '{}diff_model_{}.pt'.format(self.args.ckpt_directory, epoch + 1)
                        torch.save(self.model_diff.state_dict(), checkpoint_path_diff)
                if self.args.figure_path:
                    create_dir(self.args.figure_path)
                    with torch.no_grad():
                        self.model_drift.eval()
                        if self.model_diff:
                            self.model_diff.eval()
                        samples = self.sampler.sample(10000, 100, args.dim)
                        figure_path = '{}{}.png'.format(self.args.figure_path, epoch + 1)
                        plt = plot_samples(self.manifold, self.target, samples) if self.target is not None \
                            else plot_samples(samples, samples)
                        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        if self.args.model_directory:
            #save trained model
            create_dir(self.args.model_directory)
            checkpoint_path_drift = '{}drift_model.pt'.format(self.args.model_directory)
            torch.save(self.model_drift.state_dict(), checkpoint_path_drift)
            if self.model_diff:
                checkpoint_path_diff = '{}diff_model.pt'.format(self.args.model_directory)
                torch.save(self.model_diff.state_dict(), checkpoint_path_diff)

def run(args):
    global train_device
    train_device = args.device
    device = args.device
    n = args.dim

    data_path = args.data_path
    numpy_data = np.load(data_path)
    target = torch.from_numpy(numpy_data).to(device=device, dtype=args.dtype)
    if args.manifold == 'torus':
        manifold = Torus
    elif args.manifold == 'special_orthogonal':
        manifold = SOn
    elif args.manifold == 'unitary':
        manifold = Un
    else:
        raise not NotImplementedError('Manifolds only supported for Torus, SO(n), and U(n)...')
    dataset = TensorDataset(target)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    beta_scheduler = Beta_Scheduler(beta_s = args.beta_s, beta_f = args.beta_f)
    trainer = Trainer(args, manifold, train_loader, target, beta_scheduler)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'eval':
        drift_path = '{}drift_model_350.pt'.format(args.model_directory)
        if args.manifold == 'unitary' or args.manifold == 'special_orthogonal':
            diff_path = '{}diff_model_350.pt'.format(args.model_directory)
        else:
            diff_path = None
        trainer.model_drift.load_state_dict(torch.load(drift_path, weights_only=True, map_location=torch.device(args.device)))
        if diff_path:
            trainer.model_diff.load_state_dict(torch.load(diff_path, weights_only=True, map_location=torch.device(args.device)))
        
        samples = trainer.sampler.sample(10000, 100, args.dim)
        if args.eval_type == 'log_likelihood':
            llh = loglikelihood(samples, mul=args.dim)
            print('loglikehood is : {}'.format(llh))
        else:
            c2st_score = C2ST(target, samples, manifold)
            print('C2ST Score is: {}'.format(c2st_score))
        plt = plot_samples(manifold, target, samples)
        create_dir(args.figure_path)
        figure_path = '{}{}_{}.png'.format(args.figure_path, args.manifold, args.dim)
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    else:
        raise NotImplementedError('Either train or evaluation modes needed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--act', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--beta_s', type=float, default=0.1)
    parser.add_argument('--beta_f', type=float, default=1.0)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--ckpt_directory', type=str, default=None)
    parser.add_argument('--figure_path', type=str, default=None)
    parser.add_argument('--model_directory', type=str, default=None)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--eval_type', type=str, default='log_likelihood', choices=['log_likelihood', 'C2ST'])
    parser.add_argument('--data_type', type=str, required=True, choices=['torch.float32', 'torch.complex64'])
    parser.add_argument('--model_type', type=str, required=True, choices=['MLP', 'ResNet'])
    parser.add_argument('--manifold', type=str, required=True, choices=['torus', 'special_orthogonal', 'unitary'])

    args = parser.parse_args()
    seed = load_seed(args.seed)
    dtype = getattr(torch, args.data_type.split('.')[1])
    args.dtype = dtype
    run(args)
