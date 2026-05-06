import math
from itertools import chain

import torch
import torch.optim as optim


class SS_SOAP(optim.Optimizer):
    """
    Implements SS-SOAP with complex-safe eigenspace operations.

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01):
            Weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner basis.
        max_precond_dim (`int`, *optional*, defaults to 10000):
            Maximum dimension of the preconditioner.
        merge_dims (`bool`, *optional*, defaults to `False`):
            Whether or not to merge dimensions of the preconditioner.
        precondition_1d (`bool`, *optional*, defaults to `False`):
            Whether or not to precondition 1D gradients.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer.
        data_format (`str`, *optional*, defaults to `channels_first`):
            Data format of the input for convolutional layers.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias correction in Adam.
        downscale_method (`str`, *optional*, defaults to `"fixed"`):
            Method to downscale `exp_avg_sq` when the basis is refreshed.
            Options: `"none"`, `"fixed"`, `"frob_norm"`, `"offdiag_ratio"`.
        preconditioner_mode (`str`, *optional*, defaults to `"full"`):
            Preconditioner mode for this parameter group.
            Options: `"full"` and `"self_scaling_only"`.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
        downscale_method: str = "fixed",
        preconditioner_mode: str = "full",
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
            "downscale_method": downscale_method,
            "preconditioner_mode": preconditioner_mode,
        }
        super().__init__(params, defaults)
        self._data_format = data_format
        self._global_step = 0

    @staticmethod
    def _second_moment_dtype(dtype):
        if dtype == torch.complex32:
            return torch.float16
        if dtype == torch.complex64:
            return torch.float32
        if dtype == torch.complex128:
            return torch.float64
        return dtype

    @staticmethod
    def _linalg_work_dtype(dtype):
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        if dtype == torch.complex32:
            return torch.complex64
        return dtype

    @staticmethod
    def _fallback_linalg_dtype(dtype):
        if dtype in (torch.float64, torch.complex128):
            return None
        if dtype in (torch.complex32, torch.complex64, torch.complex128):
            return torch.complex128
        return torch.float64

    @staticmethod
    def _adjoint(mat):
        return mat.conj().transpose(-2, -1)

    @staticmethod
    def _squared_magnitude(tensor, out_dtype=None):
        if torch.is_complex(tensor):
            squared = (tensor * tensor.conj()).real
        else:
            squared = tensor.square()
        if out_dtype is not None and squared.dtype != out_dtype:
            squared = squared.to(out_dtype)
        return squared

    def _prepare_linalg_tensor(self, tensor):
        work_dtype = self._linalg_work_dtype(tensor.dtype)
        work_tensor = tensor.detach()
        if work_tensor.dtype != work_dtype:
            work_tensor = work_tensor.to(work_dtype)
        return work_tensor, tensor.dtype, tensor.device

    @staticmethod
    def _uses_full_preconditioner(source):
        if isinstance(source, dict):
            mode = source.get("preconditioner_mode", "full")
        else:
            mode = getattr(source, "preconditioner_mode", "full")
        return mode == "full"

    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions
        is less than or equal to max_precond_dim.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)

        shape = grad.shape
        new_shape = []
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        return grad.reshape(new_shape)

    def _project_with_basis(
        self,
        grad,
        state,
        merge_dims=False,
        max_precond_dim=10000,
        restore_shape=True,
    ):
        original_shape = grad.shape
        permuted_shape = None
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat.conj(), dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims and restore_shape:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)

        return grad

    def _project_back_with_basis(
        self,
        grad,
        state,
        merge_dims=False,
        max_precond_dim=10000,
        restore_shape=True,
    ):
        original_shape = grad.shape
        permuted_shape = None
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims and restore_shape:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)

        return grad

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None if closure is None else closure()

        self._global_step += 1
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                state["preconditioner_mode"] = group.get("preconditioner_mode", "full")

                if "exp_avg" not in state:
                    state["last_grad"] = torch.zeros_like(grad)
                    state["last_param"] = torch.zeros_like(p)
                    state["curr_param"] = torch.zeros_like(p)
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(
                        grad, dtype=self._second_moment_dtype(grad.dtype)
                    )

                exp_avg_sq_dtype = self._second_moment_dtype(grad.dtype)
                if torch.is_complex(state["exp_avg_sq"]) or state["exp_avg_sq"].dtype != exp_avg_sq_dtype:
                    state["exp_avg_sq"] = state["exp_avg_sq"].real.to(exp_avg_sq_dtype)

                state["curr_param"] = p.detach().clone()

                if "Q" not in state:
                    if self._uses_full_preconditioner(group):
                        self.init_preconditioner(
                            grad,
                            state,
                            precondition_frequency=group["precondition_frequency"],
                            precondition_1d=group["precondition_1d"],
                            shampoo_beta=(
                                group["shampoo_beta"]
                                if group["shampoo_beta"] >= 0
                                else group["betas"][1]
                            ),
                            max_precond_dim=group["max_precond_dim"],
                            merge_dims=group["merge_dims"],
                        )
                        self.update_preconditioner(
                            grad,
                            state,
                            max_precond_dim=group["max_precond_dim"],
                            merge_dims=group["merge_dims"],
                            precondition_1d=group["precondition_1d"],
                            downscale_method=group["downscale_method"],
                        )
                    else:
                        state["GG"] = []
                        state["Q"] = []
                        state["D"] = []
                    continue

                grad_projected = self.project(
                    grad,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                grad_sq = self._squared_magnitude(grad_projected, out_dtype=exp_avg_sq.dtype)
                exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=(1.0 - beta2))

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * (bias_correction2**0.5) / bias_correction1

                norm_grad = self.project_back(
                    exp_avg / denom,
                    state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                if group["normalize_grads"]:
                    if torch.is_complex(norm_grad):
                        norm_factor = torch.sqrt(torch.mean(torch.abs(norm_grad) ** 2)) + 1e-30
                    else:
                        norm_factor = torch.sqrt(torch.mean(norm_grad**2)) + 1e-30
                    norm_grad = norm_grad / norm_factor

                if norm_grad.dtype != p.dtype:
                    norm_grad = norm_grad.to(p.dtype)

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                self.update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                    downscale_method=group["downscale_method"],
                )

        return loss

    def init_preconditioner(
        self,
        grad,
        state,
        precondition_frequency=10,
        shampoo_beta=0.95,
        max_precond_dim=10000,
        precondition_1d=False,
        merge_dims=False,
    ):
        """
        Initializes the preconditioner matrices.
        """
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype)
                )
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))

        state["Q"] = None
        state["D"] = []
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        return self._project_with_basis(
            grad,
            state,
            merge_dims=merge_dims,
            max_precond_dim=max_precond_dim,
            restore_shape=True,
        )

    def _compute_eigenvalues(self, state):
        state["D"] = []
        for q_mat, gg_mat in zip(state["Q"], state["GG"]):
            if isinstance(q_mat, torch.Tensor) and q_mat.numel() > 0:
                state["D"].append(torch.diagonal(self._adjoint(q_mat) @ gg_mat @ q_mat).real.clamp_min(1e-12))
            else:
                state["D"].append([])

    def _align_eigenbasis(self, q_old, q_new):
        overlap = torch.diagonal(self._adjoint(q_old) @ q_new)
        phases = torch.ones_like(overlap)
        nonzero = overlap.abs() > 0
        phases[nonzero] = overlap[nonzero] / overlap[nonzero].abs()
        return q_new * phases.conj().unsqueeze(0)

    def _downscale_factor_from_overlap(self, q_old, q_new, ratio, method):
        if method == "fixed":
            return 0.25
        if method == "frob_norm":
            overlap = self._adjoint(q_old) @ q_new
            identity = torch.eye(overlap.shape[0], dtype=overlap.dtype, device=overlap.device)
            frob_norm = torch.linalg.norm(overlap - identity).item()
            return math.exp(-2.0 * frob_norm / math.sqrt(2 * overlap.shape[0]))
        if method == "offdiag_ratio":
            if ratio > 0.8:
                return 0.25
            if ratio > 0.5:
                return 0.5
            return 0.75
        return 1.0

    def update_preconditioner(
        self,
        grad,
        state,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
        downscale_method="fixed",
    ):
        """
        Updates the preconditioner matrices and the eigenbases.
        """
        if not self._uses_full_preconditioner(state):
            return

        if state["Q"] is not None:
            state["exp_avg"] = self._project_back_with_basis(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
                restore_shape=True,
            )

        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                if torch.is_complex(grad):
                    outer_product = grad.unsqueeze(1) @ grad.conj().unsqueeze(0)
                else:
                    outer_product = grad.unsqueeze(1) @ grad.unsqueeze(0)
                state["GG"][0].lerp_(outer_product, 1 - state["shampoo_beta"])
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for idx, sh in enumerate(grad.shape):
                if sh <= max_precond_dim:
                    outer_product = torch.tensordot(
                        grad,
                        grad.conj() if torch.is_complex(grad) else grad,
                        dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )
                    state["GG"][idx].lerp_(outer_product, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
            self._compute_eigenvalues(state)

        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            if downscale_method == "none":
                state["Q"] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
                self._compute_eigenvalues(state)
            else:
                new_q = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
                ratios = []
                downscale_factors = []
                recomputed = False

                for idx, (gg_mat, q_old, q_new) in enumerate(zip(state["GG"], state["Q"], new_q)):
                    if len(gg_mat) == 0:
                        ratios.append(0.0)
                        downscale_factors.append(1.0)
                        continue

                    current_basis_gg = self._adjoint(q_old) @ gg_mat @ q_old
                    diag = torch.diag(torch.diagonal(current_basis_gg))
                    offdiag_norm = torch.linalg.norm(current_basis_gg - diag)
                    full_norm = torch.linalg.norm(current_basis_gg).clamp_min(1e-12)
                    ratio = float((offdiag_norm / full_norm).real)
                    ratios.append(ratio)

                    if ratio > 0.7:
                        q_new = self._align_eigenbasis(q_old, q_new)
                        state["Q"][idx] = q_new
                        recomputed = True
                        downscale_factors.append(
                            self._downscale_factor_from_overlap(q_old, q_new, ratio, downscale_method)
                        )
                    else:
                        downscale_factors.append(1.0)

                if recomputed:
                    state["last_eig_step"] = state["step"]
                    downscale_factor = min(downscale_factors)
                    state["downscale_ema"] = state.get("downscale_ema", 1.0) * 0.9 + downscale_factor * 0.1
                    state["exp_avg_sq"].mul_(state["downscale_ema"])
                    self._compute_eigenvalues(state)

        if state["step"] > 0:
            state["exp_avg"] = self.project(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

    def _calculate_tau_self_scaling_only(self, grad, state):
        if not all(key in state for key in ["last_grad", "last_param", "curr_param"]):
            return 1.0

        param_delta = state["curr_param"] - state["last_param"]
        grad_delta = grad - state["last_grad"]

        num = (grad_delta.conj() * param_delta).real.sum().to(torch.float64)
        if not torch.isfinite(num) or num <= 0:
            return 1.0

        denom = self._squared_magnitude(param_delta, out_dtype=torch.float64).sum().clamp_min(1e-12)
        if not torch.isfinite(denom):
            return 1.0

        tau = float((num / denom).item())
        if not math.isfinite(tau) or tau <= 0:
            return 1.0

        return min(1.0, tau)

    def calculate_tau(self, grad, state, merge_dims=False, max_precond_dim=10000):
        if not self._uses_full_preconditioner(state):
            return self._calculate_tau_self_scaling_only(grad, state)

        if not all(key in state for key in ["Q", "D", "last_grad", "last_param", "curr_param"]):
            return 1.0

        if not any(isinstance(d, torch.Tensor) and d.numel() > 0 for d in state["D"]):
            return 1.0

        param_delta = state["curr_param"] - state["last_param"]
        grad_delta = grad - state["last_grad"]

        restore_shape = not merge_dims
        param_proj = self._project_with_basis(
            param_delta,
            state,
            merge_dims=merge_dims,
            max_precond_dim=max_precond_dim,
            restore_shape=restore_shape,
        )
        grad_proj = self._project_with_basis(
            grad_delta,
            state,
            merge_dims=merge_dims,
            max_precond_dim=max_precond_dim,
            restore_shape=restore_shape,
        )

        num = (grad_proj.conj() * param_proj).real.sum().to(torch.float64)
        if not torch.isfinite(num) or num <= 0:
            return 1.0

        denom = self._squared_magnitude(param_proj, out_dtype=torch.float64)
        for axis, eigvals in enumerate(state["D"]):
            if isinstance(eigvals, torch.Tensor) and eigvals.numel() > 0:
                shape = [1] * denom.ndim
                shape[axis] = eigvals.shape[0]
                denom = denom * eigvals.to(device=denom.device, dtype=denom.dtype).reshape(shape)

        denom = denom.sum().clamp_min(1e-12)
        if not torch.isfinite(denom):
            return 1.0

        tau = float((num / denom).item())
        if not math.isfinite(tau) or tau <= 0:
            return 1.0

        return min(1.0, tau)

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient back to the original space and applies self-scaling.
        """
        grad = self._project_back_with_basis(
            grad,
            state,
            merge_dims=merge_dims,
            max_precond_dim=max_precond_dim,
            restore_shape=True,
        )

        if state["step"] > 1:
            tau = self.calculate_tau(
                grad,
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )
        else:
            tau = 1.0

        tau = max(min(float(tau), 1.0), 1e-3)
        state["last_grad"] = grad.clone()
        state["last_param"] = state["curr_param"].clone()

        return grad * tau

    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh.
        """
        final = []
        eye_cache = {}

        for m in mat:
            if len(m) == 0:
                final.append([])
                continue

            work_m, original_dtype, original_device = self._prepare_linalg_tensor(m)
            eye_key = (work_m.shape[0], work_m.dtype, work_m.device)
            if eye_key not in eye_cache:
                eye_cache[eye_key] = torch.eye(
                    work_m.shape[0], dtype=work_m.dtype, device=work_m.device
                )

            try:
                _, q = torch.linalg.eigh(work_m + 1e-30 * eye_cache[eye_key])
            except RuntimeError:
                fallback_dtype = self._fallback_linalg_dtype(work_m.dtype)
                if fallback_dtype is None:
                    raise
                fallback_eye = torch.eye(
                    work_m.shape[0], dtype=fallback_dtype, device=work_m.device
                )
                _, q = torch.linalg.eigh(work_m.to(fallback_dtype) + 1e-30 * fallback_eye)

            q = torch.flip(q, [1]).to(device=original_device, dtype=original_dtype)
            final.append(q)

        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """
        Computes the eigenbases of the preconditioner using one round of power
        iteration followed by torch.linalg.qr decomposition.
        """
        matrix = []
        orth_matrix = []
        metadata = []
        for m, o in zip(state["GG"], state["Q"]):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                metadata.append(None)
                continue

            work_m, original_dtype, original_device = self._prepare_linalg_tensor(m)
            matrix.append(work_m)
            orth_matrix.append(o.detach().to(device=work_m.device, dtype=work_m.dtype))
            metadata.append((original_dtype, original_device))

        orig_shape = state["exp_avg_sq"].shape
        if self._data_format == "channels_last" and len(orig_shape) == 4:
            permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
        else:
            permuted_shape = None

        if merge_dims:
            exp_avg_sq = self.merge_dims(state["exp_avg_sq"], max_precond_dim)
        else:
            exp_avg_sq = state["exp_avg_sq"]

        final = []
        for ind, (m, o, meta) in enumerate(zip(matrix, orth_matrix, metadata)):
            if len(m) == 0:
                final.append([])
                continue

            est_eig = torch.diagonal(self._adjoint(o) @ m @ o).real
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:, sort_idx]

            try:
                q, _ = torch.linalg.qr(m @ o)
            except RuntimeError:
                fallback_dtype = self._fallback_linalg_dtype(m.dtype)
                if fallback_dtype is None:
                    raise
                q, _ = torch.linalg.qr(m.to(fallback_dtype) @ o.to(fallback_dtype))

            original_dtype, original_device = meta
            final.append(q.to(device=original_device, dtype=original_dtype))

        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state["exp_avg_sq"] = exp_avg_sq
        return final
