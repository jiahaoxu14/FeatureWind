import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.autograd.functional import jacobian as autograd_jacobian

from .mds_torch import distance_matrix_HD_tensor, mds
from .tsne import tsne, tsne_1step


def _coerce_param_mapping(params):
    mapping = {}
    if not params:
        return mapping
    if isinstance(params, dict):
        for key, value in params.items():
            mapping[str(key).strip().lower()] = value
        return mapping

    for entry in params:
        text = str(entry).strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
            mapping[key.strip().lower()] = value.strip()
            continue
        if "perplexity" not in mapping and "perp" not in mapping:
            try:
                mapping["perplexity"] = float(text)
                continue
            except Exception:
                pass
        if "quality" not in mapping and text.lower() in {"draft", "balanced", "final"}:
            mapping["quality"] = text.lower()
            continue
        if "cache" not in mapping and text.lower() in {"auto", "off", "refresh"}:
            mapping["cache"] = text.lower()
            continue
        if "seed" not in mapping:
            try:
                mapping["seed"] = int(text)
            except Exception:
                continue
    return mapping


def _quality_preset(quality):
    quality_name = str(quality or "balanced").strip().lower()
    if quality_name == "draft":
        return {
            "name": "draft",
            "max_iter": 300,
            "early_stop_config": {
                "enabled": True,
                "min_iter": 150,
                "check_interval": 5,
                "lookback_iters": 25,
                "rel_tol": 2e-3,
                "patience": 3,
            },
        }
    if quality_name == "final":
        return {
            "name": "final",
            "max_iter": 1000,
            "early_stop_config": {
                "enabled": False,
                "min_iter": 1000,
                "check_interval": 10,
                "lookback_iters": 0,
                "rel_tol": 0.0,
                "patience": 1,
            },
        }
    return {
        "name": "balanced",
        "max_iter": 1000,
        "early_stop_config": {
            "enabled": True,
            "min_iter": 250,
            "check_interval": 10,
            "lookback_iters": 50,
            "rel_tol": 5e-4,
            "patience": 3,
        },
    }


class ProjectionRunner:
    def __init__(self, projection, params=None):
        self.params = params
        self.projection = projection
        self.firstRun = False
        self.jacobian = None
        self.jacobian_numpy = None
        self.run_stats = {}
        self.timings = {}
        self.cache_info = {}

    def calculateValues(self, points, perturbations=None):
        del perturbations
        self.points = points
        self.timings = {}
        self.cache_info = {"base_state_hit": False}

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using device: CUDA")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using device: MPS")
        else:
            device = torch.device("cpu")
            print("Using device: CPU")

        data = torch.tensor(points, dtype=torch.float32, device=device, requires_grad=True)
        n_points, n_features = data.shape
        param_map = _coerce_param_mapping(self.params)

        if self.projection == tsne or (isinstance(self.projection, str) and self.projection.lower() == "tsne"):
            perp_raw = param_map.get("perplexity", param_map.get("perp", 40.0))
            try:
                perp_override = float(perp_raw)
            except Exception:
                perp_override = 40.0
            perp_base = perp_override if perp_override > 0 else 40.0
            perp_grad = perp_base

            try:
                seed = int(param_map.get("seed", 0))
            except Exception:
                seed = 0

            quality = _quality_preset(param_map.get("quality", "balanced"))
            cache_policy = str(param_map.get("cache", "auto")).strip().lower()
            if cache_policy not in {"auto", "off", "refresh"}:
                cache_policy = "auto"
            jacobian_strategy = str(param_map.get("jacobian_strategy", "auto")).strip().lower()
            if jacobian_strategy not in {"auto", "full", "chunked", "serial"}:
                jacobian_strategy = "auto"

            base_state_cache_path = param_map.get("base_state_cache_path")
            cache_path = Path(base_state_cache_path) if base_state_cache_path else None
            self.run_stats = {
                "projection": "tsne",
                "perplexity": float(perp_base),
                "quality": quality["name"],
                "seed": int(seed),
                "cache_policy": cache_policy,
                "jacobian_strategy_requested": jacobian_strategy,
            }

            init_Y = None
            init_iY = None
            init_beta = None
            step1_stats = None

            print(
                f"Step 1/3: Computing base t-SNE projection ({quality['max_iter']} iteration cap, quality={quality['name']})..."
            )
            t0 = time.time()
            if cache_path and cache_policy == "auto" and cache_path.exists():
                loaded = self._load_base_state_cache(cache_path, device)
                if loaded is not None:
                    init_Y, init_iY, init_beta, step1_stats = loaded
                    self.cache_info["base_state_hit"] = True
                    print(f"  Loaded base t-SNE state from cache: {cache_path}")
            if init_Y is None:
                with torch.no_grad():
                    Y_base, params, step1_stats = tsne(
                        data,
                        2,
                        quality["max_iter"],
                        10,
                        perp_base,
                        save_params=True,
                        seed=seed,
                        early_stop_config=quality["early_stop_config"],
                        return_stats=True,
                    )
                del Y_base
                init_Y, init_iY, init_beta = params
                if cache_path and cache_policy != "off":
                    self._save_base_state_cache(cache_path, init_Y, init_iY, init_beta, step1_stats)
            self.timings["step1_seconds"] = float(time.time() - t0)
            self.run_stats["base_state"] = step1_stats or {}
            print(f"  Step 1 done in {self.timings['step1_seconds']:.1f}s")

            print("Step 2/3: Computing 1-step projection...")
            t1 = time.time()
            with torch.no_grad():
                Y = tsne_1step(
                    data.detach(),
                    init_Y,
                    init_iY,
                    perplexity=perp_grad,
                    init_beta=init_beta,
                )
            self.timings["step2_seconds"] = float(time.time() - t1)
            print(f"  Step 2 done in {self.timings['step2_seconds']:.1f}s")

            print(f"Step 3/3: Computing Jacobian for {n_points} points...")
            t2 = time.time()
            grads_x, grads_y, strategy_used = self._compute_jacobian_tsne(
                data=data,
                init_Y=init_Y,
                init_iY=init_iY,
                init_beta=init_beta,
                perplexity=perp_grad,
                device=device,
                n_points=n_points,
                n_features=n_features,
                strategy=jacobian_strategy,
            )
            self.timings["step3_seconds"] = float(time.time() - t2)
            self.timings["total_seconds"] = float(time.time() - t0)
            self.run_stats["jacobian_strategy_used"] = strategy_used
            print(
                f"  Step 3 done in {self.timings['step3_seconds']:.1f}s  (total: {self.timings['total_seconds']:.1f}s)"
            )

        elif self.projection == mds or (isinstance(self.projection, str) and self.projection.lower() == "mds"):
            t0 = time.time()
            self.run_stats = {"projection": "mds"}
            print("Step 1/3: Computing base MDS projection (999 iterations)...")
            with torch.no_grad():
                dist_hd = distance_matrix_HD_tensor(data)
                Y_base = mds(dist_hd, n_components=2, max_iter=999)
            self.timings["step1_seconds"] = float(time.time() - t0)

            print("Step 2/3: Computing projection with gradients (1 iteration)...")
            t1 = time.time()
            dist_hd = distance_matrix_HD_tensor(data)
            Y = mds(dist_hd, n_components=2, max_iter=1)
            del Y_base
            self.timings["step2_seconds"] = float(time.time() - t1)

            print(f"Step 3/3: Computing Jacobian for {n_points} points (serial)...")
            t2 = time.time()

            def mds_fn(x):
                return mds(distance_matrix_HD_tensor(x), n_components=2, max_iter=1)

            grads_x, grads_y = self._compute_jacobian_serial(data, mds_fn, n_points, n_features, device)
            self.timings["step3_seconds"] = float(time.time() - t2)
            self.timings["total_seconds"] = float(time.time() - t0)
            self.run_stats["jacobian_strategy_used"] = "serial"
            print(
                f"  Step 3 done in {self.timings['step3_seconds']:.1f}s  (total: {self.timings['total_seconds']:.1f}s)"
            )
        else:
            raise ValueError("Unsupported projection method. Use 'tsne' or 'mds'.")

        self.outPoints = Y
        self.grads = torch.stack([grads_x, grads_y], dim=1).detach().cpu().numpy()

        self.jacobian = torch.zeros(2 * n_points, n_features, dtype=torch.float32, device=device)
        self.jacobian[0::2] = grads_x
        self.jacobian[1::2] = grads_y
        self.jacobian_numpy = self.jacobian.detach().cpu().numpy()

        print("✓ Tangent map generation completed successfully!")

    def _load_base_state_cache(self, cache_path, device):
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                init_Y = torch.tensor(cached["init_Y"], dtype=torch.float32, device=device)
                init_iY = torch.tensor(cached["init_iY"], dtype=torch.float32, device=device)
                init_beta = torch.tensor(cached["init_beta"], dtype=torch.float32, device=device)
                stats_json = str(cached["run_stats_json"].tolist())
            return init_Y, init_iY, init_beta, json.loads(stats_json)
        except Exception as exc:
            print(f"  Failed to load base-state cache ({exc}); recomputing.")
            return None

    def _save_base_state_cache(self, cache_path, init_Y, init_iY, init_beta, run_stats):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                init_Y=init_Y.detach().cpu().numpy().astype(np.float32, copy=False),
                init_iY=init_iY.detach().cpu().numpy().astype(np.float32, copy=False),
                init_beta=init_beta.detach().cpu().numpy().astype(np.float32, copy=False),
                run_stats_json=np.asarray(json.dumps(run_stats or {})),
            )
        except Exception as exc:
            print(f"  Failed to save base-state cache ({exc}).")

    def _compute_jacobian_tsne(
        self,
        data,
        init_Y,
        init_iY,
        init_beta,
        perplexity,
        device,
        n_points,
        n_features,
        strategy="auto",
    ):
        def tsne_1step_fn(x):
            return tsne_1step(x, init_Y, init_iY, perplexity=perplexity, init_beta=init_beta)

        estimated_bytes = int(n_points) * 2 * int(n_points) * int(n_features) * 4
        use_full = strategy == "full" or (
            strategy == "auto" and n_points <= 512 and estimated_bytes <= 1_000_000_000
        )
        if use_full:
            try:
                J = autograd_jacobian(tsne_1step_fn, data, create_graph=False, vectorize=True)
                idx = torch.arange(n_points, device=device)
                grads_x = J[idx, 0, idx, :]
                grads_y = J[idx, 1, idx, :]
                print("  (used full vectorized Jacobian)")
                return grads_x, grads_y, "full"
            except Exception as exc:
                if strategy == "full":
                    print(f"  Full Jacobian failed ({exc}), falling back to serial loop...")
                    grads_x, grads_y = self._compute_jacobian_serial(data, tsne_1step_fn, n_points, n_features, device)
                    return grads_x, grads_y, "serial-fallback"
                print(f"  Full Jacobian unavailable ({exc}); trying chunked diagonal blocks...")

        if strategy in {"auto", "chunked"}:
            try:
                grads_x, grads_y = self._compute_jacobian_tsne_chunked(
                    data,
                    tsne_1step_fn,
                    n_points=n_points,
                    n_features=n_features,
                    device=device,
                    chunk_size=64,
                )
                print("  (used chunked diagonal Jacobian)")
                return grads_x, grads_y, "chunked"
            except Exception as exc:
                print(f"  Chunked Jacobian failed ({exc}), falling back to serial loop...")

        grads_x, grads_y = self._compute_jacobian_serial(data, tsne_1step_fn, n_points, n_features, device)
        return grads_x, grads_y, "serial-fallback"

    def _compute_jacobian_tsne_chunked(self, data, fn, n_points, n_features, device, chunk_size=64):
        Y = fn(data)
        grads_x = torch.zeros(n_points, n_features, device=device)
        grads_y = torch.zeros(n_points, n_features, device=device)
        for start in range(0, n_points, chunk_size):
            end = min(n_points, start + chunk_size)
            batch_indices = torch.arange(start, end, device=device)
            batch_size = int(end - start)
            for coord, target in ((0, grads_x), (1, grads_y)):
                grad_outputs = torch.zeros(batch_size, n_points, 2, dtype=Y.dtype, device=device)
                grad_outputs[torch.arange(batch_size, device=device), batch_indices, coord] = 1.0
                batched = torch.autograd.grad(
                    outputs=Y,
                    inputs=data,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    is_grads_batched=True,
                    allow_unused=False,
                )[0]
                target[start:end] = batched[torch.arange(batch_size, device=device), batch_indices, :]
        return grads_x, grads_y

    def _compute_jacobian_serial(self, data, fn, n_points, n_features, device):
        Y = fn(data)
        grads_x = torch.zeros(n_points, n_features, device=device)
        grads_y = torch.zeros(n_points, n_features, device=device)
        progress_interval = max(1, n_points // 10)
        for i in range(n_points):
            if i % progress_interval == 0 or i == n_points - 1:
                print(f"  Computing gradients: {i + 1}/{n_points} ({(i + 1) / n_points * 100:.0f}%)")
            grads_x[i] = torch.autograd.grad(Y[i, 0], data, retain_graph=True)[0][i]
            grads_y[i] = torch.autograd.grad(Y[i, 1], data, retain_graph=True)[0][i]
        return grads_x, grads_y

    def get_jacobian_for_point(self, point_idx):
        if self.jacobian is None:
            raise ValueError("Jacobian not computed. Run calculateValues first.")
        return self.jacobian_numpy[2 * point_idx : 2 * point_idx + 2, :]

    def compute_metric_tensor(self, point_idx):
        J_i = self.get_jacobian_for_point(point_idx)
        return np.dot(J_i.T, J_i)

    def pushforward_vector(self, point_idx, high_d_vector):
        J_i = self.get_jacobian_for_point(point_idx)
        return np.dot(J_i, high_d_vector)

    def metric_normalized_pushforward(self, point_idx, high_d_vector):
        J_i = self.get_jacobian_for_point(point_idx)
        v_2d = np.dot(J_i, high_d_vector)
        G = self.compute_metric_tensor(point_idx)
        metric_scale = np.sqrt(np.trace(G))
        if metric_scale > 1e-10:
            v_2d = v_2d / metric_scale
        return v_2d


projections = ["tsne", "mds", "Tangent-Map"]
projectionClasses = [tsne, mds, None]
projectionParamOpts = [["Perplexity", "Max_Iterations", "Number_of_Dimensions"], []]


def readFile(filename):
    with open(filename, "rt") as handle:
        read = csv.reader(handle)
        points = []
        firstLine = next(read)
        rowDat = []
        head = False
        for i in range(0, len(firstLine)):
            try:
                rowDat.append(float(firstLine[i]))
            except Exception:
                head = True
                break
        if not head:
            points.append(rowDat)

        for row in read:
            rowDat = []
            for i in range(0, len(row)):
                try:
                    rowDat.append(float(row[i]))
                except Exception:
                    print("invalid data type - must be numeric")
                    exit(0)
            points.append(rowDat)
    return points
