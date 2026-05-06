import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb
import os
import sys
import pickle
from time import time
from scipy.integrate import odeint

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from scripts.optimizers.muon import SingleDeviceMuonWithAuxAdam
    from scripts.optimizers.soap import SOAP
    from scripts.optimizers.self_scaled_soap import SS_SOAP
    from scripts.optimizers.purifying_soap import Purifying_SOAP
except ImportError:
    pass


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return self.out(x)

class PirateNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1, layers=4):
        super().__init__()
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_encoder_z = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        u = self.act(self.input_encoder(x))
        v = self.act(self.input_encoder_z(x))
        for layer in self.layers:
            z = self.act(layer(u))
            u = (1 - z) * u + z * v 
        return self.out(u)

# physics, reference solver
def solve_ac_spectral(nx=200, nt=100):
    """
    Solves the Allen-Cahn equation using a Spectral (FFT) method to serve as 
    the Ground Truth reference solution.
    u_t = 0.0001*u_xx - 5*u^3 + 5*u
    """
    print("Computing high-fidelity reference solution via Spectral Method...")
    
    # domain [-1, 1]
    L = 2.0
    x = np.linspace(-1, 1, nx, endpoint=False) # Periodic, so endpoint excluded for FFT
    t = np.linspace(0, 1, nt)
    
    # IC: u(0, x) = x^2 * cos(pi * x)
    u0 = x**2 * np.cos(np.pi * x)
    
    # wavenumbers for FFT
    # k = 2*pi*n / L
    k = 2 * np.pi * np.fft.fftfreq(nx, d=L/nx)
    
    # ODE function for scipy.odeint
    # FT(u_t) = -0.0001 * k^2 * FT(u) + FT(5u - 5u^3)
    def ac_rhs(u_hat_complex_flattened, t, k, epsilon=0.0001):
        # Reconstruct complex array
        u_hat = u_hat_complex_flattened[:nx] + 1j * u_hat_complex_flattened[nx:]
        
        # linear term (diffusion) in fourier space
        # u_xx -> (ik)^2 u_hat = -k^2 u_hat
        diff_term = -epsilon * (k**2) * u_hat

        # nonlinear term in physical space
        u = np.fft.ifft(u_hat).real
        nonlin = 5 * u - 5 * u**3
        nonlin_hat = np.fft.fft(nonlin)

        # total RHS
        rhs_hat = diff_term + nonlin_hat
        
        return np.concatenate([rhs_hat.real, rhs_hat.imag])     # flatten back to real for odeint

    # IC in fourier space
    u0_hat = np.fft.fft(u0)
    u0_hat_flat = np.concatenate([u0_hat.real, u0_hat.imag])
    
    u_hat_sol = odeint(ac_rhs, u0_hat_flat, t, args=(k,))   # time integration
    
    # transform back to physical space
    u_ref = np.zeros((nt, nx))
    for i in range(nt):
        u_h = u_hat_sol[i, :nx] + 1j * u_hat_sol[i, nx:]
        u_ref[i, :] = np.fft.ifft(u_h).real

    # add the endpoint back for plotting consistency (u(-1) = u(1))
    # we generated data on [-1, 1), plot on [-1, 1]
    u_ref_final = np.hstack([u_ref, u_ref[:, 0:1]])
    
    print("Reference solution computed.")
    return u_ref_final # Shape (nt, nx+1)

def sample_ac_points(n_collocation, n_init, n_boundary, device):
    """
    Sample points for training.
    """
    t_f = torch.rand(n_collocation, 1, device=device)
    x_f = torch.rand(n_collocation, 1, device=device) * 2 - 1
    X_f = torch.cat([t_f, x_f], dim=1).requires_grad_(True)

    x_init = torch.rand(n_init, 1, device=device) * 2 - 1
    t_init = torch.zeros(n_init, 1, device=device)
    X_init = torch.cat([t_init, x_init], dim=1)
    u_init = x_init**2 * torch.cos(np.pi * x_init)

    t_bc = torch.rand(n_boundary, 1, device=device)
    x_bc_left = torch.ones(n_boundary, 1, device=device) * -1
    x_bc_right = torch.ones(n_boundary, 1, device=device) * 1
    X_bc_left = torch.cat([t_bc, x_bc_left], dim=1).requires_grad_(True)
    X_bc_right = torch.cat([t_bc, x_bc_right], dim=1).requires_grad_(True)

    return X_f, X_init, u_init, X_bc_left, X_bc_right

def compute_loss(model, X_f, X_init, u_init, X_bc_left, X_bc_right):
    u = model(X_f)
    u_g = grad(u.sum(), X_f, create_graph=True)[0]
    u_t, u_x = u_g[:, 0:1], u_g[:, 1:2]
    u_xx = grad(u_x.sum(), X_f, create_graph=True)[0][:, 1:2]

    # u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0
    f = u_t - 0.0001 * u_xx + 5 * u**3 - 5 * u
    loss_f = torch.mean(f**2)

    u_pred_init = model(X_init)
    loss_init = torch.mean((u_pred_init - u_init)**2)

    u_left = model(X_bc_left)
    u_right = model(X_bc_right)
    u_x_left = grad(u_left.sum(), X_bc_left, create_graph=True)[0][:, 1:2]
    u_x_right = grad(u_right.sum(), X_bc_right, create_graph=True)[0][:, 1:2]
    loss_bc = torch.mean((u_left - u_right)**2) + torch.mean((u_x_left - u_x_right)**2)

    return loss_f, loss_init, loss_bc


def save_plots(model, plot_dir, epoch, device, u_ref):
    """
    Plots solution heatmap AND absolute error heatmap.
    u_ref: numpy array of shape (nt, nx) corresponding to the meshgrid below.
    """
    model.eval()
    
    # Define grid to match u_ref shape (100, 201)
    # u_ref was generated with nt=100, nx=200 + 1 endpoint = 201
    nt_plot = 100
    nx_plot = 201 
    
    t = np.linspace(0, 1, nt_plot)
    x = np.linspace(-1, 1, nx_plot)
    tt, xx = np.meshgrid(t, x) # Shapes: (nx, nt)
    
    # Flatten for model prediction
    # Note: meshgrid default is 'xy' indexing, so tt is (nx, nt).
    # We usually want X_star to be list of (t, x) points.
    X_star = np.hstack((tt.flatten()[:, None], xx.flatten()[:, None]))
    X_star_torch = torch.tensor(X_star, dtype=torch.float64, device=device)
    
    with torch.no_grad():
        u_pred = model(X_star_torch)
        # Reshape back to (nx, nt) then transpose to (nt, nx) for imshow
        # Prediction shape: (201*100, 1) -> (201, 100) -> Transpose to (100, 201)
        u_pred = u_pred.cpu().numpy().reshape(nx_plot, nt_plot).T

    # --- 1. Plot Prediction ---
    fig1 = plt.figure(figsize=(9, 5))
    plt.imshow(u_pred, interpolation='nearest', cmap='rainbow', 
               extent=[0, 1, -1, 1], origin='lower', aspect='auto')
    plt.colorbar(label='u(t,x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'Prediction at Epoch {epoch}')
    
    plt.savefig(os.path.join(plot_dir, f"solution_ep{epoch}.pdf"))
    plt.savefig(os.path.join(plot_dir, f"solution_ep{epoch}.png"), dpi=100)
    plt.close(fig1)

    # --- 2. Plot Absolute Error ---
    if u_ref is not None:
        # Ensure shapes match
        if u_pred.shape == u_ref.shape:
            abs_error = np.abs(u_pred - u_ref)
            
            fig2 = plt.figure(figsize=(9, 5))
            plt.imshow(abs_error, interpolation='nearest', cmap='inferno', 
                       extent=[0, 1, -1, 1], origin='lower', aspect='auto')
            plt.colorbar(label='|Error|')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title(f'Absolute Error at Epoch {epoch}\nMax Error: {abs_error.max():.2e}')
            
            plt.savefig(os.path.join(plot_dir, f"error_ep{epoch}.pdf"))
            plt.savefig(os.path.join(plot_dir, f"error_ep{epoch}.png"), dpi=100)
            plt.close(fig2)

            if wandb.run is not None:
                wandb.log({
                    "Prediction Heatmap": wandb.Image(os.path.join(plot_dir, f"solution_ep{epoch}.png")),
                    "Error Heatmap": wandb.Image(os.path.join(plot_dir, f"error_ep{epoch}.png")),
                    "L2 Relative Error": np.linalg.norm(abs_error) / np.linalg.norm(u_ref)
                }, step=epoch)
        else:
            print(f"Shape mismatch: Pred {u_pred.shape} vs Ref {u_ref.shape}. Skipping error plot.")

    model.train()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Allen-Cahn PINN with SS-SOAP")
    
    # Experiment Config
    parser.add_argument('--run', type=int, default=1, help='Run ID')
    parser.add_argument('--arch', type=str, default="pirate", choices=["mlp", "pirate"], help="Network architecture")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000000)
    parser.add_argument('--plot_every', type=int, default=500)
    
    # Optimizer Config
    parser.add_argument('--optim', type=str, default="SS-SOAP", 
                        choices=["Adam", "SingleDeviceMuonWithAuxAdam", "SOAP", "SS-SOAP", "Purifying_SOAP"])
    
    # Hyperparams
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam LR')
    parser.add_argument('--muon_lr', type=float, default=0.02)
    parser.add_argument('--soap_lr', type=float, default=3e-3)
    parser.add_argument('--shampoo_beta', type=float, default=-1)
    parser.add_argument('--precondition_frequency', type=int, default=10)
    parser.add_argument('--soap_rank', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0)
    
    # Purifying SOAP specifics
    parser.add_argument('--eig_check_interval', type=int, default=10)
    parser.add_argument('--eig_trigger_tau', type=float, default=0.1)
    parser.add_argument('--eig_warmup', type=int, default=50)
    parser.add_argument('--eig_cool_down', type=int, default=10)

    # Weights for losses
    parser.add_argument('--w_pde', type=float, default=1.0)
    parser.add_argument('--w_init', type=float, default=100.0)
    parser.add_argument('--w_bc', type=float, default=100.0)

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.run)
    np.random.seed(args.run)

    # WandB
    wandb.init(project="AllenCahn_PINN_Optimization", config=vars(args), name=f"{args.optim}_{args.arch}_run{args.run}")
    
    plot_dir = f"./ac_plots/AC_{args.optim}_{args.arch}_run{args.run}"
    os.makedirs(plot_dir, exist_ok=True)

    # --- Generate Reference Solution (Once) ---
    # We generate on a 200x100 grid. The spectral solver takes nx=200 points.
    # It returns (100, 201) because we appended the periodic endpoint.
    u_ref = solve_ac_spectral(nx=200, nt=100)

    # Model Initialization
    if args.arch == "mlp":
        model = MLP(input_dim=2, hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    else:
        model = PirateNet(input_dim=2, hidden_dim=args.hidden_dim, layers=args.layers).to(device)

    # Optimizer Initialization
    params = model.parameters()
    
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "SOAP":
        optimizer = SOAP(params, lr=args.soap_lr, betas=(0.9, 0.95), 
                         shampoo_beta=args.shampoo_beta, weight_decay=args.weight_decay,
                         precondition_frequency=args.precondition_frequency)
    elif args.optim == "SS-SOAP":
        optimizer = SS_SOAP(params, lr=args.soap_lr, betas=(0.9, 0.95),
                            shampoo_beta=args.shampoo_beta, weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency)
    elif args.optim == "Purifying_SOAP":
        optimizer = Purifying_SOAP(params, lr=args.soap_lr, betas=(0.9, 0.95),
                                   shampoo_beta=args.shampoo_beta, weight_decay=args.weight_decay,
                                   precondition_frequency=args.precondition_frequency,
                                   eig_check_interval=args.eig_check_interval,
                                   eig_trigger_tau=args.eig_trigger_tau,
                                   eig_warmup=args.eig_warmup,
                                   eig_cool_down=args.eig_cool_down)
    elif args.optim == "SingleDeviceMuonWithAuxAdam":
        muon_params = [p for p in params if p.ndim >= 2]
        adam_params = [p for p in params if p.ndim < 2]
        param_groups = [
            {'params': muon_params, 'lr': args.muon_lr, 'use_muon': True, 'momentum': 0.95},
            {'params': adam_params, 'lr': args.lr, 'use_muon': False, 'betas': (0.9, 0.95)}
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Data Sampling
    X_f, X_init, u_init, X_bc_left, X_bc_right = sample_ac_points(20000, 2000, 2000, device)

    print(f"Starting training with {args.optim} on {args.arch} for Allen-Cahn...")
    
    metrics = {'loss_pde': [], 'loss_init': [], 'loss_bc': [], 'loss_total': []}
    
    start_time = time()

    for ep in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss_f, loss_init, loss_bc = compute_loss(model, X_f, X_init, u_init, X_bc_left, X_bc_right)
        loss = args.w_pde * loss_f + args.w_init * loss_init + args.w_bc * loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        metrics['loss_pde'].append(loss_f.item())
        metrics['loss_init'].append(loss_init.item())
        metrics['loss_bc'].append(loss_bc.item())
        metrics['loss_total'].append(loss.item())

        if ep % 100 == 0:
            elapsed = time() - start_time
            print(f"Ep {ep}/{args.epochs} | Loss: {loss.item():.2e} | PDE: {loss_f.item():.2e} | Init: {loss_init.item():.2e} | BC: {loss_bc.item():.2e} | Time: {elapsed:.1f}s")
            
            log_dict = {
                "epoch": ep,
                "Total Loss": loss.item(),
                "PDE Loss": loss_f.item(),
                "Init Loss": loss_init.item(),
                "BC Loss": loss_bc.item(),
                "LR": scheduler.get_last_lr()[0]
            }
            if hasattr(optimizer, 'eigen_similarities') and optimizer.eigen_similarities:
                log_dict["Eigen Similarity"] = np.mean(optimizer.eigen_similarities[-1])
            if hasattr(optimizer, 'anisotropy_ratios') and optimizer.anisotropy_ratios:
                log_dict["Anisotropy Ratio"] = np.mean(optimizer.anisotropy_ratios[-1])

            wandb.log(log_dict, step=ep)

        # Plotting
        if ep % args.plot_every == 0:
            # Pass u_ref to save_plots
            save_plots(model, plot_dir, ep, device, u_ref)

    with open(os.path.join(plot_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    
    torch.save(model.state_dict(), os.path.join(plot_dir, "model_final.pth"))
    wandb.finish()
    print("Training Complete.")

if __name__ == "__main__":
    main()
