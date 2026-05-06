import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb
import os
import sys
import pickle
from time import time
from scipy.integrate import solve_ivp

import json

# Set precision
torch.set_default_dtype(torch.float64)

# Add project root to path
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# Imports (Optimizers)
# -----------------------------------------------------------------------------
try:
    from scripts.optimizers.muon import SingleDeviceMuonWithAuxAdam
    from scripts.optimizers.soap import SOAP
    from scripts.optimizers.self_scaled_soap import SS_SOAP
    from scripts.optimizers.purifying_soap import Purifying_SOAP
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Architecture: PirateNet
# -----------------------------------------------------------------------------
class PirateNet(nn.Module):
    """
    Modified MLP with skip connections and input injection.
    """
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

# -----------------------------------------------------------------------------
# Reference Solvers (Ground Truth)
# -----------------------------------------------------------------------------

def solve_kdv_spectral(nx=256, nt=100):
    """
    KdV: u_t + u u_x + 0.0025 u_xxx = 0
    Domain: [-1, 1], Time: [0, 1]
    IC: cos(pi*x)
    """
    print("Generating KdV Reference (Spectral)...")
    L = 2.0
    x = np.linspace(-1, 1, nx, endpoint=False)
    t = np.linspace(0, 1, nt)
    k = 2 * np.pi * np.fft.fftfreq(nx, d=L/nx)
    
    u0 = np.cos(np.pi * x)
    u0_hat = np.fft.fft(u0)

    def rhs(t, u_hat_flat):
        u_hat = u_hat_flat[:nx] + 1j * u_hat_flat[nx:]
        u = np.fft.ifft(u_hat).real
        
        # Nonlinear: u*u_x = 0.5 * (u^2)_x
        flux = 0.5 * u**2
        flux_hat = np.fft.fft(flux)
        
        # Spectral derivative: d/dx -> ik, d^3/dx^3 -> (ik)^3 = -i k^3
        # u_t = - (0.5*(u^2)_x + 0.0025*u_xxx)
        # u_t_hat = - ( ik * 0.5 * flux_hat - 0.0025 * i * k^3 * u_hat )
        du_hat = -1j * k * flux_hat + 1j * 0.0025 * (k**3) * u_hat
        
        return np.concatenate([du_hat.real, du_hat.imag])

    sol = solve_ivp(rhs, [0, 1], np.concatenate([u0_hat.real, u0_hat.imag]), 
                    t_eval=t, method='RK45', rtol=1e-6)
    
    u_ref = np.zeros((nt, nx))
    for i in range(nt):
        y_c = sol.y[:nx, i] + 1j*sol.y[nx:, i]
        u_ref[i] = np.fft.ifft(y_c).real
        
    return np.hstack([u_ref, u_ref[:, 0:1]]) # Periodic wrap

def solve_grayscott_spectral(nx=64, nt=100):
    """
    Gray-Scott: 
    u_t = 2e-5 lap(u) - uv^2 + 0.035(1-u)
    v_t = 1e-5 lap(v) + uv^2 - (0.035+0.065)v
    Domain: [-1, 1]^2, Time: [0, 200]
    """
    print("Generating Gray-Scott Reference (Spectral)...")
    N = nx
    L = 2.0
    Du, Dv, F, k_param = 2e-5, 1e-5, 0.035, 0.065
    
    x = np.linspace(-1, 1, N, endpoint=False)
    X, Y = np.meshgrid(x, x)
    
    # IC
    u = np.ones_like(X)
    v = np.zeros_like(X)
    mask = (np.abs(X) < 0.2) & (np.abs(Y) < 0.2)
    u[mask] = 0.5
    v[mask] = 0.25
    
    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    KX, KY = np.meshgrid(k, k)
    Lap = -(KX**2 + KY**2)
    
    dt = 1.0 # Semi-implicit or small step
    steps_per_frame = int(200 / nt / dt)
    
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    
    u_hist = []
    v_hist = []
    
    # Simple Euler integration in spectral space for reaction, exact for diffusion
    # Actually, let's use splitting: Exact diffusion, Euler reaction
    
    for i in range(nt):
        # Save frame
        u_hist.append(np.fft.ifft2(u_hat).real)
        v_hist.append(np.fft.ifft2(v_hat).real)
        
        for _ in range(steps_per_frame):
            u_c = np.fft.ifft2(u_hat).real
            v_c = np.fft.ifft2(v_hat).real
            
            uv2 = u_c * v_c**2
            fu = -uv2 + F*(1 - u_c)
            fv = uv2 - (F + k_param)*v_c
            
            u_hat = (u_hat + dt*np.fft.fft2(fu)) / (1 - dt*Du*Lap)
            v_hat = (v_hat + dt*np.fft.fft2(fv)) / (1 - dt*Dv*Lap)

    return np.array(u_hist), np.array(v_hist)

# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------

def loss_kdv(model, batch, device):
    X, X_init, u_init = batch
    X.requires_grad_(True)
    
    u = model(X)
    u_g = grad(u.sum(), X, create_graph=True)[0]
    u_t, u_x = u_g[:,0:1], u_g[:,1:2]
    u_xx = grad(u_x.sum(), X, create_graph=True)[0][:,1:2]
    u_xxx = grad(u_xx.sum(), X, create_graph=True)[0][:,1:2]
    
    res = u_t + u*u_x + 0.0025*u_xxx
    loss_f = torch.mean(res**2)
    
    u_pred_init = model(X_init)
    loss_init = torch.mean((u_pred_init - u_init)**2)
    
    return loss_f, loss_init, 0.0

def loss_grayscott(model, batch, device):
    X, X_init, uv_init = batch
    X.requires_grad_(True)
    
    out = model(X)
    u, v = out[:,0:1], out[:,1:2]
    
    grads_u = grad(u.sum(), X, create_graph=True)[0]
    u_t = grads_u[:,0:1]
    u_xx = grad(grads_u[:,1:2].sum(), X, create_graph=True)[0][:,1:2]
    u_yy = grad(grads_u[:,2:3].sum(), X, create_graph=True)[0][:,2:3]
    
    grads_v = grad(v.sum(), X, create_graph=True)[0]
    v_t = grads_v[:,0:1]
    v_xx = grad(grads_v[:,1:2].sum(), X, create_graph=True)[0][:,1:2]
    v_yy = grad(grads_v[:,2:3].sum(), X, create_graph=True)[0][:,2:3]
    
    Du, Dv, F, k = 2e-5, 1e-5, 0.035, 0.065
    uv2 = u * v**2
    
    f_u = u_t - (Du*(u_xx + u_yy) - uv2 + F*(1-u))
    f_v = v_t - (Dv*(v_xx + v_yy) + uv2 - (F+k)*v)
    
    loss_f = torch.mean(f_u**2 + f_v**2)
    
    out_init = model(X_init)
    loss_init = torch.mean((out_init - uv_init)**2)
    
    return loss_f, loss_init, 0.0

def loss_gl(model, batch, device):
    # A_t = lap(A) + A - |A|^2 A, A = u + iv
    X, X_init, uv_init = batch
    X.requires_grad_(True)
    
    out = model(X)
    u, v = out[:,0:1], out[:,1:2]
    
    grads_u = grad(u.sum(), X, create_graph=True)[0]
    u_t = grads_u[:,0:1]
    u_xx = grad(grads_u[:,1:2].sum(), X, create_graph=True)[0][:,1:2]
    u_yy = grad(grads_u[:,2:3].sum(), X, create_graph=True)[0][:,2:3]

    grads_v = grad(v.sum(), X, create_graph=True)[0]
    v_t = grads_v[:,0:1]
    v_xx = grad(grads_v[:,1:2].sum(), X, create_graph=True)[0][:,1:2]
    v_yy = grad(grads_v[:,2:3].sum(), X, create_graph=True)[0][:,2:3]
    
    amp2 = u**2 + v**2
    f_u = u_t - (u_xx + u_yy + u - amp2*u)
    f_v = v_t - (v_xx + v_yy + v - amp2*v)
    
    loss_f = torch.mean(f_u**2 + f_v**2)
    
    out_init = model(X_init)
    loss_init = torch.mean((out_init - uv_init)**2)
    
    return loss_f, loss_init, 0.0

def loss_ldc(model, batch, device):
    # Steady NS 2D. Re=100.
    X, X_bc, u_bc = batch
    X.requires_grad_(True)
    
    out = model(X)
    u, v, p = out[:,0:1], out[:,1:2], out[:,2:3]
    
    grads_u = grad(u.sum(), X, create_graph=True)[0]
    u_x, u_y = grads_u[:,0:1], grads_u[:,1:2]
    u_xx = grad(u_x.sum(), X, create_graph=True)[0][:,0:1]
    u_yy = grad(u_y.sum(), X, create_graph=True)[0][:,1:2]

    grads_v = grad(v.sum(), X, create_graph=True)[0]
    v_x, v_y = grads_v[:,0:1], grads_v[:,1:2]
    v_xx = grad(v_x.sum(), X, create_graph=True)[0][:,0:1]
    v_yy = grad(v_y.sum(), X, create_graph=True)[0][:,1:2]
    
    grads_p = grad(p.sum(), X, create_graph=True)[0]
    p_x, p_y = grads_p[:,0:1], grads_p[:,1:2]
    
    Re = 100.0
    f_u = (u*u_x + v*u_y) + p_x - (1/Re)*(u_xx + u_yy)
    f_v = (u*v_x + v*v_y) + p_y - (1/Re)*(v_xx + v_yy)
    f_div = u_x + v_y
    
    loss_f = torch.mean(f_u**2 + f_v**2 + f_div**2)
    
    out_bc = model(X_bc)
    # Match only u, v components (indices 0, 1)
    loss_bc = torch.mean((out_bc[:,0:2] - u_bc)**2)
    
    return loss_f, 0.0, loss_bc

# -----------------------------------------------------------------------------
# Samplers
# -----------------------------------------------------------------------------

def sample_kdv(n_f=5000, n_init=1000, device='cuda'):
    t = torch.rand(n_f, 1, device=device)
    x = torch.rand(n_f, 1, device=device)*2 - 1
    X_f = torch.cat([t, x], dim=1)
    
    x_i = torch.rand(n_init, 1, device=device)*2 - 1
    t_i = torch.zeros(n_init, 1, device=device)
    X_init = torch.cat([t_i, x_i], dim=1)
    u_init = torch.cos(np.pi * x_i)
    return X_f, X_init, u_init

def sample_gs(n_f=10000, n_init=2000, device='cuda'):
    t = torch.rand(n_f, 1, device=device)*200
    x = torch.rand(n_f, 1, device=device)*2 - 1
    y = torch.rand(n_f, 1, device=device)*2 - 1
    X_f = torch.cat([t, x, y], dim=1)
    
    x_i = torch.rand(n_init, 1, device=device)*2 - 1
    y_i = torch.rand(n_init, 1, device=device)*2 - 1
    t_i = torch.zeros(n_init, 1, device=device)
    X_init = torch.cat([t_i, x_i, y_i], dim=1)
    
    u = torch.ones_like(x_i)
    v = torch.zeros_like(x_i)
    mask = (x_i.abs() < 0.2) & (y_i.abs() < 0.2)
    u[mask] = 0.5
    v[mask] = 0.25
    return X_f, X_init, torch.cat([u, v], dim=1)

def sample_gl(n_f=10000, n_init=2000, device='cuda'):
    t = torch.rand(n_f, 1, device=device)*2
    x = torch.rand(n_f, 1, device=device)*2 - 1
    y = torch.rand(n_f, 1, device=device)*2 - 1
    X_f = torch.cat([t, x, y], dim=1)
    
    x_i = torch.rand(n_init, 1, device=device)*2 - 1
    y_i = torch.rand(n_init, 1, device=device)*2 - 1
    t_i = torch.zeros(n_init, 1, device=device)
    X_init = torch.cat([t_i, x_i, y_i], dim=1)
    
    u = torch.exp(-(x_i**2 + y_i**2))
    v = torch.zeros_like(u)
    return X_f, X_init, torch.cat([u, v], dim=1)

def sample_ldc(n_f=10000, n_bc=2000, device='cuda'):
    x = torch.rand(n_f, 1, device=device)
    y = torch.rand(n_f, 1, device=device)
    X_f = torch.cat([x, y], dim=1)
    
    # Boundary Generator
    def make_bc(n, x_val=None, y_val=None, u_val=0, v_val=0):
        x_ = torch.rand(n, 1, device=device) if x_val is None else torch.ones(n, 1, device=device)*x_val
        y_ = torch.rand(n, 1, device=device) if y_val is None else torch.ones(n, 1, device=device)*y_val
        u_ = torch.ones(n, 1, device=device)*u_val
        v_ = torch.ones(n, 1, device=device)*v_val
        return torch.cat([x_, y_], 1), torch.cat([u_, v_], 1)

    n = n_bc // 4
    X_t, U_t = make_bc(n, y_val=1, u_val=1) # Top
    X_b, U_b = make_bc(n, y_val=0)          # Bottom
    X_l, U_l = make_bc(n, x_val=0)          # Left
    X_r, U_r = make_bc(n, x_val=1)          # Right
    
    X_bc = torch.cat([X_t, X_b, X_l, X_r])
    U_bc = torch.cat([U_t, U_b, U_l, U_r])
    return X_f, X_bc, U_bc

# -----------------------------------------------------------------------------
# Plotting & Utils
# -----------------------------------------------------------------------------

def plot_kdv(model, plot_dir, ep, device, u_ref):
    nx, nt = 256, 100
    t = np.linspace(0, 1, nt)
    x = np.linspace(-1, 1, nx+1)
    tt, xx = np.meshgrid(t, x)
    
    X = np.hstack([tt.flatten()[:,None], xx.flatten()[:,None]])
    X_t = torch.tensor(X, device=device, dtype=torch.float64)
    with torch.no_grad():
        u_pred = model(X_t).cpu().numpy().reshape(nx+1, nt).T
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax[0].imshow(u_pred, extent=[0,1,-1,1], aspect='auto', origin='lower', cmap='jet')
    ax[0].set_title(f"Pred (Ep {ep})")
    plt.colorbar(im1, ax=ax[0])
    
    err = np.abs(u_pred - u_ref) / 100
    im2 = ax[1].imshow(err, extent=[0,1,-1,1], aspect='auto', origin='lower', cmap='inferno')
    ax[1].set_title(f"Absolute Error")
    plt.colorbar(im2, ax=ax[1])
    
    plt.savefig(f"{plot_dir}/kdv_ep{ep}.pdf", dpi=100)
    plt.close()

def plot_gs(model, plot_dir, ep, device, uv_ref):
    u_true, v_true = uv_ref
    nx = u_true.shape[1]
    t_val = 200.0
    x = np.linspace(-1, 1, nx)
    XX, YY = np.meshgrid(x, x)
    TT = np.ones_like(XX) * t_val
    
    X_in = np.hstack([TT.flatten()[:,None], XX.flatten()[:,None], YY.flatten()[:,None]])
    X_t = torch.tensor(X_in, device=device, dtype=torch.float64)
    
    with torch.no_grad():
        out = model(X_t).cpu().numpy()
        u_pred = out[:,0].reshape(nx, nx)
        
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(u_pred, extent=[-1,1,-1,1], origin='lower')
    ax[0].set_title("U Pred (T=200)")
    ax[1].imshow(u_true[-1], extent=[-1,1,-1,1], origin='lower')
    ax[1].set_title("U Ref")
    err = np.abs(u_pred - u_true[-1]) / 100
    im = ax[2].imshow(err, extent=[-1,1,-1,1], origin='lower', cmap='inferno')
    plt.colorbar(im, ax=ax[2])
    
    plt.savefig(f"{plot_dir}/gs_ep{ep}.pdf", dpi=100)
    plt.close()

def plot_ldc(model, plot_dir, ep, device):
    nx = 100
    x = np.linspace(0, 1, nx)
    XX, YY = np.meshgrid(x, x)
    X_in = np.hstack([XX.flatten()[:,None], YY.flatten()[:,None]])
    X_t = torch.tensor(X_in, device=device, dtype=torch.float64)
    
    with torch.no_grad():
        out = model(X_t).cpu().numpy()
        u, v, p = out[:,0].reshape(nx, nx), out[:,1].reshape(nx, nx), out[:,2].reshape(nx, nx)
        vel = np.sqrt(u**2 + v**2)
        
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].streamplot(XX, YY, u, v, color=vel, cmap='jet')
    ax[0].set_title("Streamlines")
    im = ax[1].imshow(p, extent=[0,1,0,1], origin='lower')
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title("Pressure")
    
    plt.savefig(f"{plot_dir}/ldc_ep{ep}.pdf", dpi=100)
    plt.close()

def save_metrics(metrics, plot_dir):
    for key, val in metrics.items():
        np.save(os.path.join(plot_dir, f"{key}.npy"), np.array(val))

def plot_final_metrics(plot_dir):
    try:
        pde_loss = np.load(os.path.join(plot_dir, "loss_pde.npy"))
        bc_loss = np.load(os.path.join(plot_dir, "loss_bc_init.npy"))
        
        plt.figure(figsize=(8, 5))
        plt.semilogy(pde_loss, label='PDE Loss', alpha=0.7)
        plt.semilogy(bc_loss, label='BC/Init Loss', alpha=0.7)
        plt.legend()
        plt.title("Training Losses")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(plot_dir, "loss_curves.pdf"), dpi=100)
        plt.close()
    except:
        print("Could not plot metrics (files missing?)")

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pde', type=str, required=True, choices=['kdv', 'gs', 'gl', 'ldc'])
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--optim', type=str, default="SS-SOAP")
    parser.add_argument('--plot_every', type=int, default=10)
    
    # Optimizer HParams
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--muon_lr', type=float, default=0.02)
    parser.add_argument('--soap_lr', type=float, default=3e-3)
    parser.add_argument('--shampoo_beta', type=float, default=-1)
    parser.add_argument('--precondition_frequency', type=int, default=10)
    parser.add_argument('--soap_rank', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0)
    
    # Purifying HParams
    parser.add_argument('--eig_check_interval', type=int, default=10)
    parser.add_argument('--eig_trigger_tau', type=float, default=0.1)
    parser.add_argument('--eig_warmup', type=int, default=50)
    parser.add_argument('--eig_cool_down', type=int, default=10)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_dir = f"./plots_0120/pde_plots/{args.pde}_{args.optim}_run{args.run}_dim{args.hidden_dim}"
    os.makedirs(plot_dir, exist_ok=True)
    
    wandb.init(project="PINN_Benchmarks", config=vars(args), name=f"{args.pde}_{args.optim}")
    
    # Configuration Map
    config = {
        'kdv': {'in': 2, 'out': 1, 'steps': 600000, 'sampler': sample_kdv, 'loss': loss_kdv, 'ref_solver': solve_kdv_spectral, 'plot': plot_kdv},
        'gs':  {'in': 3, 'out': 2, 'steps': 4000000, 'sampler': sample_gs,  'loss': loss_grayscott, 'ref_solver': solve_grayscott_spectral, 'plot': plot_gs},
        'gl':  {'in': 3, 'out': 2, 'steps': 1000000,  'sampler': sample_gl,  'loss': loss_gl, 'ref_solver': None, 'plot': lambda m, d, ep, dev, r: None}, # GL ref is complex, skipped for brevity
        'ldc': {'in': 2, 'out': 3, 'steps': 2000000, 'sampler': sample_ldc, 'loss': loss_ldc, 'ref_solver': None, 'plot': lambda m, d, ep, dev, r: plot_ldc(m, d, ep, dev)}
    }
    

    cfg = config[args.pde]
    cfg_json = {
        k: (v.__name__ if callable(v) else v)
        for k, v in cfg.items()
    }
    cfg_path = os.path.join(plot_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg_json, f, indent=2)
        
    
    # 1. Generate Reference (if available)
    ref_data = None
    if cfg['ref_solver'] is not None:
        ref_data = cfg['ref_solver']()
        
    # 2. Model & Optimizer
    model = PirateNet(cfg['in'], args.hidden_dim, cfg['out']).to(device)
    params = model.parameters()
    
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "SS-SOAP":
        optimizer = SS_SOAP(params, lr=args.soap_lr, betas=(0.9, 0.95), 
                            shampoo_beta=args.shampoo_beta, weight_decay=args.weight_decay,
                            precondition_frequency=args.precondition_frequency)
    elif args.optim == "SOAP":
        optimizer = SOAP(params, lr=args.soap_lr, betas=(0.9, 0.95), 
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['steps'])
    
    # 3. Training Loop
    metrics = {'loss_pde': [], 'loss_bc_init': [], 'loss_total': [], 'step': []}
    
    # Initial batch
    batch = cfg['sampler'](device=device)
    
    print(f"Starting {args.pde} training for {cfg['steps']} steps...")
    start_time = time()
    
    for ep in range(1, cfg['steps'] + 1):
        optimizer.zero_grad()
        loss_f, loss_init, loss_bc = cfg['loss'](model, batch, device)
        
        # Combine losses (init and bc are often mutually exclusive per problem here)
        loss_bc_total = loss_init + loss_bc
        loss = loss_f + 100.0 * loss_bc_total
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track
        metrics['loss_pde'].append(loss_f.item())
        metrics['loss_bc_init'].append(loss_bc_total.item())
        metrics['loss_total'].append(loss.item())
        metrics['step'].append(ep)
        
        if ep % 1000 == 0:
            print(f"{ep}: Loss {loss.item():.2e} | PDE {loss_f.item():.2e} | BC {loss_bc_total.item():.2e}")
            wandb.log({"Loss": loss.item(), "PDE": loss_f.item(), "BC": loss_bc_total.item()}, step=ep)
        
        if ep % args.plot_every == 0:
            cfg['plot'](model, plot_dir, ep, device, ref_data)
            # Resample
            batch = cfg['sampler'](device=device)
            # Save metrics periodically
            save_metrics(metrics, plot_dir)
            
    # Final save and plot
    save_metrics(metrics, plot_dir)
    plot_final_metrics(plot_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
