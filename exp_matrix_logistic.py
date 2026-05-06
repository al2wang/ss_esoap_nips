import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pickle
from torch.autograd.functional import hessian

# Add project root
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from scripts.optimizers.soap_track_eigenvec import SOAP
    from scripts.optimizers.self_scaled_soap import SS_SOAP
    from scripts.optimizers.purifying_soap import Purifying_SOAP
    from scripts.optimizers.muon import SingleDeviceMuonWithAuxAdam  # Added Muon import
except ImportError:
    print("Warning: Custom optimizers not found. Ensure paths are correct.")

def get_optimizer(args, params):
    if args.optim == 'Adam':
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    elif args.optim == 'Muon':
        # Muon Logic: 2D+ params use Muon, <2D use Adam
        weight_params = [p for p in params if p.ndim >= 2]
        bias_params = [p for p in params if p.ndim < 2]
        
        param_groups = [
            {
                "params": weight_params,
                "lr": args.muon_lr,
                "momentum": 0.95, # Standard Muon momentum
                "weight_decay": args.weight_decay,
                "use_muon": True
            },
            {
                "params": bias_params,
                "lr": args.lr,
                "betas": args.betas,
                "weight_decay": args.weight_decay,
                "use_muon": False
            }
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)

    elif args.optim == 'SOAP':
        return SOAP(params, lr=args.soap_lr, betas=args.betas, shampoo_beta=args.shampoo_beta,
                    weight_decay=args.weight_decay, precondition_frequency=args.precondition_frequency,
                    max_precond_dim=args.soap_rank, normalize_grads=False, correct_bias=True)
    elif args.optim == 'SS-SOAP':
        return SS_SOAP(params, lr=args.soap_lr, betas=args.betas, shampoo_beta=args.shampoo_beta,
                       weight_decay=args.weight_decay, precondition_frequency=args.precondition_frequency,
                       max_precond_dim=args.soap_rank, normalize_grads=False, correct_bias=True,
                       downscale_method="fixed")
    elif args.optim == 'Purifying_SOAP':
        return Purifying_SOAP(params, lr=args.soap_lr, betas=args.betas, shampoo_beta=args.shampoo_beta,
                              weight_decay=args.weight_decay, precondition_frequency=args.precondition_frequency,
                              max_precond_dim=args.soap_rank, normalize_grads=False, correct_bias=True,
                              eig_check_interval=args.eig_check_interval, eig_trigger_tau=args.eig_trigger_tau,
                              eig_warmup=args.eig_warmup, eig_cool_down=args.eig_cool_down)
    else:
        raise ValueError(f"Optimizer {args.optim} not recognized")

def compute_condition_number(loss_fn, u_param):
    # Computes Hessian Condition Number
    def func(u):
        return loss_fn(u)
    H = hessian(func, u_param)
    d, r = u_param.shape
    H_flat = H.reshape(d*r, d*r)
    try:
        eigvals = torch.linalg.eigvalsh(H_flat)
        eigvals = eigvals[eigvals > 1e-6]
        if len(eigvals) > 0:
            return (eigvals.max() / eigvals.min()).item()
        else:
            return 1.0
    except:
        return 1.0

def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Problem Dimensions
    d = 20
    r = 2
    m = 200 # More samples for classification

    # Ground Truth Matrix
    U_star = torch.randn(d, r, device=device)
    M_star = U_star @ U_star.T

    # Measurement Matrices
    A = torch.randn(m, d, d, device=device) / np.sqrt(d)
    
    # Generate Labels: y = sign(<A, M*>)
    inner_prods = (A * M_star.unsqueeze(0)).sum(dim=(1,2))
    y = torch.sign(inner_prods)
    # Ensure no zero labels
    y[y == 0] = 1.0

    # Initialization
    U = torch.nn.Parameter(torch.randn(d, r, device=device) * 0.01)

    optimizer = get_optimizer(args, [U])
    
    metrics = {
        'loss': [],
        'acc': [],
        'grad_nuclear_norm': [],
        'hessian_cond': [],
        'iter': []
    }

    # Determine learning rate for folder name
    if 'SOAP' in args.optim:
        lr_name = args.soap_lr
    elif 'Muon' in args.optim:
        lr_name = args.muon_lr
    else:
        lr_name = args.lr

    plot_dir = f"./plots_logistic_{args.optim}_lr{lr_name}_seed{args.seed}"
    os.makedirs(plot_dir, exist_ok=True)

    def loss_closure(u_in=None):
        u_curr = U if u_in is None else u_in
        M_pred = u_curr @ u_curr.T
        pred_inner = (A * M_pred.unsqueeze(0)).sum(dim=(1,2))
        
        # Logistic Loss: log(1 + exp(-y * pred))
        # Using softplus for numerical stability: softplus(x) = log(1 + exp(x))
        loss = F.softplus(-y * pred_inner).mean()
        return loss

    def get_acc(u_curr):
        M_pred = u_curr @ u_curr.T
        pred_inner = (A * M_pred.unsqueeze(0)).sum(dim=(1,2))
        preds = torch.sign(pred_inner)
        return (preds == y).float().mean().item()

    print(f"Starting Matrix Logistic Regression with {args.optim}...")

    for step in range(args.steps):
        optimizer.zero_grad()
        loss = loss_closure()
        loss.backward()
        
        with torch.no_grad():
            grad_nuc = torch.norm(U.grad, p='nuc').item()
            acc = get_acc(U)
            
            metrics['loss'].append(loss.item())
            metrics['acc'].append(acc)
            metrics['grad_nuclear_norm'].append(grad_nuc)
            metrics['iter'].append(step)

            if step % 10 == 0:
                cond = compute_condition_number(loss_closure, U)
                metrics['hessian_cond'].append(cond)
            else:
                metrics['hessian_cond'].append(metrics['hessian_cond'][-1] if len(metrics['hessian_cond']) > 0 else 1.0)

        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: Loss {loss.item():.6f}, Acc {acc:.2f}, Cond {metrics['hessian_cond'][-1]:.2f}")
            
            # Plotting
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            axs[0].plot(metrics['iter'], metrics['loss'])
            axs[0].set_yscale('log')
            axs[0].set_title('Logistic Loss')
            
            axs[1].plot(metrics['iter'], metrics['grad_nuclear_norm'])
            axs[1].set_yscale('log')
            axs[1].set_title('Gradient Nuclear Norm')
            
            axs[2].plot(metrics['iter'], metrics['hessian_cond'])
            axs[2].set_yscale('log')
            axs[2].set_title('Hessian Condition Number')
            
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/metrics_step_{step}.pdf")
            plt.close()

            with open(f"{plot_dir}/metrics.pkl", "wb") as f:
                pickle.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix Logistic Regression Experiment")
    
    # Added Muon to choices
    parser.add_argument('--optim', type=str, default="SS-SOAP", choices=["Adam", "SOAP", "SS-SOAP", "Purifying_SOAP", "Muon"])
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=1501)
    
    # Optimizer Hyperparams
    parser.add_argument('--lr', type=float, default=1e-3, help="Adam LR")
    parser.add_argument('--muon_lr', type=float, default=0.02, help="Muon LR") # Added Muon LR
    parser.add_argument('--soap_lr', type=float, default=0.01)
    parser.add_argument('--soap_rank', type=int, default=10)
    parser.add_argument('--shampoo_beta', type=float, default=-1)
    parser.add_argument('--betas', type=float, nargs=2, default=(0.95, 0.95))
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--precondition_frequency', type=int, default=10)
    
    # Purifying Specifics
    parser.add_argument('--eig_check_interval', type=int, default=10)
    parser.add_argument('--eig_trigger_tau', type=float, default=0.1)
    parser.add_argument('--eig_warmup', type=int, default=50)
    parser.add_argument('--eig_cool_down', type=int, default=10)

    args = parser.parse_args()
    
    run_experiment(args)
