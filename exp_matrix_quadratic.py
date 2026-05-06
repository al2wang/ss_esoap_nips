import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pickle
from torch.autograd.functional import hessian

# Add project root to path (adjust "../.." if your folder structure differs)
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Imports based on your provided structure
try:
    from scripts.optimizers.soap_track_eigenvec import SOAP
    from scripts.optimizers.self_scaled_soap import SS_SOAP
    from scripts.optimizers.purifying_soap import Purifying_SOAP
    from scripts.optimizers.muon import SingleDeviceMuonWithAuxAdam  # Added Muon import
except ImportError:
    print("Warning: Custom optimizers not found in path. Please ensure scripts/optimizers/ contains the required files.")

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
                       downscale_method="fixed") # Defaulting downscale
    elif args.optim == 'Purifying_SOAP':
        return Purifying_SOAP(params, lr=args.soap_lr, betas=args.betas, shampoo_beta=args.shampoo_beta,
                              weight_decay=args.weight_decay, precondition_frequency=args.precondition_frequency,
                              max_precond_dim=args.soap_rank, normalize_grads=False, correct_bias=True,
                              eig_check_interval=args.eig_check_interval, eig_trigger_tau=args.eig_trigger_tau,
                              eig_warmup=args.eig_warmup, eig_cool_down=args.eig_cool_down)
    else:
        raise ValueError(f"Optimizer {args.optim} not recognized")

def compute_condition_number(loss_fn, u_param):
    """
    Computes the condition number of the Hessian of the loss w.r.t U.
    WARNING: Expensive. Only feasible for small dimensions.
    """
    # functional hessian requires a function input
    def func(u):
        return loss_fn(u)
    
    H = hessian(func, u_param)
    # H will be (d, r, d, r). Flatten to (d*r, d*r)
    d, r = u_param.shape
    H_flat = H.reshape(d*r, d*r)
    
    # Compute eigenvalues
    try:
        eigvals = torch.linalg.eigvalsh(H_flat)
        # Filter small numerical noise
        eigvals = eigvals[eigvals > 1e-6]
        if len(eigvals) > 0:
            cond = eigvals.max() / eigvals.min()
            return cond.item()
        else:
            return 1.0
    except:
        return 1.0

def run_experiment(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Problem Dimensions (Small scale to allow Hessian computation)
    d = 20
    r = 2
    m = 100 # Number of measurements

    # Ground Truth
    U_star = torch.randn(d, r, device=device)
    M_star = U_star @ U_star.T

    # Measurement Operators (Gaussian)
    A = torch.randn(m, d, d, device=device) / np.sqrt(m)
    
    # Observations b = <A_i, M*>
    # <A, B> = trace(A.T @ B) = sum(A * B)
    b = (A * M_star.unsqueeze(0)).sum(dim=(1,2)) # shape (m,)

    # Initialization
    U = torch.nn.Parameter(torch.randn(d, r, device=device) * 0.1)

    optimizer = get_optimizer(args, [U])
    
    # Metrics
    metrics = {
        'loss': [],
        'residual_error': [], # ||UU^T - M*||_F
        'grad_norm': [],
        'grad_nuclear_norm': [],
        'hessian_cond': [],
        'iter': []
    }

    plot_dir = f"./plots_quadratic_{args.optim}_lr{args.soap_lr if 'SOAP' in args.optim else args.muon_lr if 'Muon' in args.optim else args.lr}_seed{args.seed}"
    os.makedirs(plot_dir, exist_ok=True)

    def loss_closure(u_in=None):
        u_curr = U if u_in is None else u_in
        M_pred = u_curr @ u_curr.T
        # Prediction: <A_i, M_pred>
        b_pred = (A * M_pred.unsqueeze(0)).sum(dim=(1,2))
        loss = 0.5 * torch.mean((b_pred - b)**2)
        return loss

    print(f"Starting Matrix Quadratic Regression with {args.optim}...")

    for step in range(args.steps):
        optimizer.zero_grad()
        loss = loss_closure()
        loss.backward()
        
        # Track metrics before step
        with torch.no_grad():
            grad_norm = U.grad.norm().item()
            grad_nuc = torch.norm(U.grad, p='nuc').item()
            M_pred = U @ U.T
            residual = torch.norm(M_pred - M_star).item()
            
            metrics['loss'].append(loss.item())
            metrics['residual_error'].append(residual)
            metrics['grad_norm'].append(grad_norm)
            metrics['grad_nuclear_norm'].append(grad_nuc)
            metrics['iter'].append(step)

            # Compute Condition Number every 10 steps (expensive)
            if step % 10 == 0:
                cond = compute_condition_number(loss_closure, U)
                metrics['hessian_cond'].append(cond)
            else:
                metrics['hessian_cond'].append(metrics['hessian_cond'][-1] if len(metrics['hessian_cond']) > 0 else 1.0)

        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: Loss {loss.item():.6f}, Cond {metrics['hessian_cond'][-1]:.6f}, Grad Nuc {grad_nuc:.6f}")
            
            # Plotting
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            
            axs[0, 0].plot(metrics['iter'], metrics['loss'])
            axs[0, 0].set_yscale('log')
            axs[0, 0].set_title('Loss')
            
            axs[0, 1].plot(metrics['iter'], metrics['residual_error'])
            axs[0, 1].set_yscale('log')
            axs[0, 1].set_title('Residual Error ||UU^T - M*||_F')
            
            axs[1, 0].plot(metrics['iter'], metrics['grad_nuclear_norm'])
            axs[1, 0].set_yscale('log')
            axs[1, 0].set_title('Gradient Nuclear Norm')
            
            axs[1, 1].plot(metrics['iter'], metrics['hessian_cond'])
            axs[1, 1].set_yscale('log')
            axs[1, 1].set_title('Hessian Condition Number')
            
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/metrics_step_{step}.pdf")
            plt.close()

            # Save data
            with open(f"{plot_dir}/metrics.pkl", "wb") as f:
                pickle.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix Quadratic Regression Experiment")
    
    # Added Muon to choices
    parser.add_argument('--optim', type=str, default="Muon", choices=["Adam", "SOAP", "SS-SOAP", "Purifying_SOAP", "Muon"])
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=4001)
    
    # Optimizer Hyperparams
    parser.add_argument('--lr', type=float, default=1e-3, help="Adam LR (and Muon bias LR)")
    parser.add_argument('--muon_lr', type=float, default=0.02, help="Muon LR") # Added Muon LR
    parser.add_argument('--soap_lr', type=float, default=0.01)
    parser.add_argument('--soap_rank', type=int, default=10) # Typically d/2 or smaller
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
