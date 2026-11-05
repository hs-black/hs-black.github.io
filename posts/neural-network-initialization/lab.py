#!/usr/bin/env python3
"""Initialization research lab: NumPy / CPU / float64, no downloads.

OPENBLAS_NUM_THREADS=1 python3 lab.py
OPENBLAS_NUM_THREADS=1 python3 lab.py --sweep --output lab-results.json

The quick run uses width 16, depths 4/12, two seeds, 20 full-batch SGD steps.
--sweep uses widths 16/32, depths 4/12, three seeds, three learning rates,
and 40 steps: 108 training runs with identical data and equal search budgets.
All schemes share the input embedding and readout initialization. The hidden
matrices differ: Xavier N(0,1/n), He N(0,2/n), or sqrt(2) times a Haar
orthogonal matrix. All biases start at zero and are trained. No normalization,
residuals, momentum, clipping, weight decay, or learning-rate schedule is used.

The full hidden Jacobian is d a_L / d a_0, an n-by-n matrix at one fixed
probe input. Its spectrum includes numerical zeros; numerical rank uses
the standard SVD threshold n * machine_epsilon * largest_singular_value.
It is separate from the scalar output head and from the parameter Jacobian.
"""

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMES = ("xavier", "he", "orthogonal")
DATA_SEED = 20261105


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def cosine(a, b):
    """Uncentered normalized inner product; undefined for a zero vector."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else None


def format_optional(value, spec=".6f"):
    """Keep undefined diagnostics visible in the text report."""
    return "NA" if value is None else format(value, spec)


def relu_correlation(c):
    c = float(np.clip(c, -1, 1))
    return float((np.sqrt(1 - c*c) + (np.pi - np.arccos(c))*c) / np.pi)


def orthogonal(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    # Positive R diagonal makes the Q factor Haar distributed.
    return q * np.where(np.diag(r) < 0, -1., 1.)


def initialize(n, depth, scheme, seed):
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(depth):
        if scheme == "orthogonal":
            weights.append(np.sqrt(2.) * orthogonal(rng, n))
        else:
            gain = 1. if scheme == "xavier" else np.sqrt(2.)
            weights.append(gain * rng.normal(size=(n, n)) / np.sqrt(n))
    head_rng = np.random.default_rng(10000 + seed)
    return {
        "weights": weights,
        "biases": [np.zeros(n) for _ in range(depth)],
        "readout": head_rng.normal(size=n) / np.sqrt(n),
        "output_bias": np.zeros(1),
    }


def parameters(model):
    return model["weights"] + model["biases"] + [model["readout"], model["output_bias"]]


def forward(model, x):
    activations, preactivations = [x], []
    for w, b in zip(model["weights"], model["biases"]):
        z = activations[-1] @ w.T + b
        preactivations.append(z)
        activations.append(np.maximum(z, 0))
    output = activations[-1] @ model["readout"] + model["output_bias"][0]
    return output, activations, preactivations


def loss_and_grad(model, x, y):
    output, acts, pre = forward(model, x)
    residual = output - y
    dout = residual / len(x)
    grad_readout = acts[-1].T @ dout
    grad_output_bias = np.array([dout.sum()])
    upstream = dout[:, None] * model["readout"]
    grad_w, grad_b = [], []
    for layer in range(len(pre) - 1, -1, -1):
        dz = upstream * (pre[layer] > 0)
        grad_w.append(dz.T @ acts[layer])
        grad_b.append(dz.sum(axis=0))
        upstream = dz @ model["weights"][layer]
    gradients = grad_w[::-1] + grad_b[::-1] + [grad_readout, grad_output_bias]
    return float(np.mean(residual**2) / 2), gradients


def hidden_jacobian(model, x):
    """Derivative of last hidden activation w.r.t. first hidden input."""
    _, _, pre = forward(model, x[None, :])
    jacobian = np.eye(len(x))
    for w, z in zip(model["weights"], pre):
        jacobian = ((z[0] > 0)[:, None] * w) @ jacobian
    return jacobian


def embedding(n):
    # Fixed 8-D task; P.T @ P = (n/8) I. Normalized input Gram does not
    # change with hidden width, and E[||P u||^2 / n] = 1 for u ~ N(0,I_8).
    q, _ = np.linalg.qr(np.random.default_rng(1701 + n).normal(size=(n, 8)))
    return q * np.sqrt(n / 8)


def dataset(n):
    rng = np.random.default_rng(DATA_SEED)
    physical = rng.normal(size=(192, 8))
    targets = np.sin(physical[:, 0]) + .3 * physical[:, 1] * physical[:, 2]
    inputs = physical @ embedding(n).T
    return inputs[:64], targets[:64], inputs[64:128], targets[64:128], inputs[128:], targets[128:]


def moments(x):
    """Pooled coordinate/sample moments, with one global empirical mean."""
    return {"mean": float(np.mean(x)), "second_moment": float(np.mean(x*x)),
            "centered_variance": float(np.var(x))}


def spectrum(jac):
    values = np.linalg.svd(jac, compute_uv=False)
    threshold = max(jac.shape) * np.finfo(float).eps * values[0]
    active = values[values > threshold]
    return {
        "singular_values": values.tolist(),
        "rank_tolerance": float(threshold),
        "numerical_rank": int(len(active)),
        "zero_count": int(len(values) - len(active)),
        "mean_squared_singular_value": float(np.mean(values**2)),
        "largest_singular_value": float(values[0]),
        "smallest_nonzero_singular_value": float(active[-1]) if len(active) else None,
    }


def initialization_diagnostics(n, depth, scheme, seed):
    model = initialize(n, depth, scheme, seed)
    x, _, valid, _, probe, _ = dataset(n)
    observations = np.concatenate([x, valid, probe])
    _, acts, pre = forward(model, observations)
    # A fixed unit-norm upstream vector isolates propagation from readout/loss.
    direction = np.random.default_rng(8080 + n).normal(size=n)
    direction /= np.linalg.norm(direction)
    upstream = np.broadcast_to(direction, acts[-1].shape).copy()
    gradients = [None] * (depth + 1)
    gradients[-1] = upstream
    for layer in range(depth - 1, -1, -1):
        upstream = (upstream * (pre[layer] > 0)) @ model["weights"][layer]
        gradients[layer] = upstream
    layer_rows = []
    predicted_moment = float(np.mean(observations**2))
    predicted_cosine = cosine(observations[0], observations[1])
    gain_squared = 1. if scheme == "xavier" else 2.
    # All rows describe the same two fixed physical inputs, sample 0 and 1.
    for layer in range(depth + 1):
        act = acts[layer]
        centered = act - np.mean(act, axis=0, keepdims=True)
        covariance = centered.T @ centered / len(act)
        cov_eigen = np.maximum(np.linalg.eigvalsh(covariance), 0)
        total = np.sum(cov_eigen)
        prob = cov_eigen[cov_eigen > 0] / total if total else np.array([])
        effective_rank = float(np.exp(-np.sum(prob*np.log(prob)))) if total else 0.
        row = {
            "layer": layer, "activation": moments(act),
            "activation_pair_cosine": cosine(act[0], act[1]),
            "centered_feature_covariance_effective_rank": effective_rank,
            "gradient_coordinate_rms": rms(gradients[layer]),
            "gradient_mean_sample_l2": float(np.mean(np.linalg.norm(gradients[layer], axis=1))),
        }
        if layer:
            row["preactivation"] = moments(pre[layer-1])
            row["preactivation_pair_cosine"] = cosine(pre[layer-1][0], pre[layer-1][1])
            row["inactive_fraction"] = float(np.mean(pre[layer-1] <= 0))
            predicted_q = gain_squared * predicted_moment
            predicted_moment = predicted_q / 2
            row["wide_gaussian_reference"] = {
                "preactivation_second_moment": predicted_q,
                "activation_second_moment": predicted_moment,
                "preactivation_pair_correlation": predicted_cosine,
                "activation_pair_correlation": relu_correlation(predicted_cosine),
            }
            predicted_cosine = relu_correlation(predicted_cosine)
        layer_rows.append(row)
    return {
        "width": n, "depth": depth, "scheme": scheme, "seed": seed,
        "layers": layer_rows,
        "hidden_jacobian_at_probe_0": spectrum(hidden_jacobian(model, probe[0])),
    }


def run_training(n, depth, scheme, seed, lr, steps):
    model = initialize(n, depth, scheme, seed)
    train_x, train_y, valid_x, valid_y, probe_x, probe_y = dataset(n)
    initial_output, initial_features, _ = forward(model, probe_x)
    initial_loss, _ = loss_and_grad(model, train_x, train_y)
    trace, failure = [], None
    checkpoints = {1, 5, 10, 20, steps}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(1, steps + 1):
            loss, gradients = loss_and_grad(model, train_x, train_y)
            if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients):
                failure = {"step": step, "reason": "nonfinite_loss_or_gradient"}
                break
            old_norms = [float(np.linalg.norm(w)) for w in model["weights"]]
            # Updates include weights and all zero-initialized biases.
            for param, grad in zip(parameters(model), gradients):
                param -= lr * grad
            if not all(np.isfinite(param).all() for param in parameters(model)):
                failure = {"step": step, "reason": "nonfinite_parameter"}
                break
            if step in checkpoints:
                output, features, _ = forward(model, probe_x)
                after_loss, _ = loss_and_grad(model, train_x, train_y)
                valid_pred = forward(model, valid_x)[0]
                stats = {
                    "step": step, "train_loss": after_loss,
                    "validation_loss": float(np.mean((valid_pred-valid_y)**2)/2),
                    "probe_loss": float(np.mean((output-probe_y)**2)/2),
                    "layer_relative_weight_update": [
                        float(lr*np.linalg.norm(g)/norm) if norm else None
                        for g, norm in zip(gradients[:depth], old_norms)],
                    "layer_absolute_bias_update": [float(lr*np.linalg.norm(g)) for g in gradients[depth:2*depth]],
                    "layer_probe_feature_drift_rms": [rms(a-b) for a, b in zip(features[1:], initial_features[1:])],
                    "probe_output_drift_rms": rms(output-initial_output),
                }
                if not np.isfinite([stats["train_loss"], stats["validation_loss"], stats["probe_loss"], stats["probe_output_drift_rms"]]).all():
                    failure = {"step": step, "reason": "nonfinite_evaluation"}
                    break
                trace.append(stats)
    success = failure is None
    return {
        "width": n, "depth": depth, "scheme": scheme, "seed": seed,
        "learning_rate": lr, "requested_steps": steps, "initial_train_loss": initial_loss,
        "completed": success, "failure": failure,
        "final_train_loss": trace[-1]["train_loss"] if success else None,
        "final_validation_loss": trace[-1]["validation_loss"] if success else None,
        "final_probe_loss": trace[-1]["probe_loss"] if success else None,
        "trace": trace,
    }


def aggregate(runs, widths, depths, rates, seeds):
    rows = []
    for n in widths:
        for depth in depths:
            for scheme in SCHEMES:
                candidates = []
                for lr in rates:
                    group = [run for run in runs if (run["width"], run["depth"], run["scheme"], run["learning_rate"]) == (n, depth, scheme, lr)]
                    complete = [run for run in group if run["completed"]]
                    losses = [run["final_validation_loss"] for run in complete]
                    candidates.append({
                        "learning_rate": lr, "runs": len(group), "failures": len(group)-len(complete),
                        "validation_loss_mean_completed": float(np.mean(losses)) if losses else None,
                        "validation_loss_std_completed": float(np.std(losses, ddof=1)) if len(losses)>1 else None,
                        "validation_loss_per_seed": [{"seed": run["seed"], "loss": run["final_validation_loss"], "completed": run["completed"]} for run in group],
                    })
                # A candidate that failed any seed cannot win selection.
                eligible = [item for item in candidates if item["failures"] == 0]
                best = min(eligible, key=lambda item: item["validation_loss_mean_completed"]) if eligible else None
                rows.append({"width": n, "depth": depth, "scheme": scheme, "candidates": candidates,
                             "selected_learning_rate": best["learning_rate"] if best else None,
                             "selected_at_search_boundary": best["learning_rate"] in (min(rates), max(rates)) if best and len(rates)>1 else None})
    return rows


def self_checks():
    rng = np.random.default_rng(42)
    # Verify actual backprop, including all types of trainable bias.
    model = initialize(5, 3, "he", 13)
    x, y = rng.normal(size=(9, 5)), rng.normal(size=9)
    _, gradients = loss_and_grad(model, x, y)
    max_error = 0.
    for param, grad in zip(parameters(model), gradients):
        for _ in range(min(5, param.size)):
            index = tuple(rng.integers(size) for size in param.shape)
            old = param[index]
            param[index] = old + 1e-6
            plus = loss_and_grad(model, x, y)[0]
            param[index] = old - 1e-6
            minus = loss_and_grad(model, x, y)[0]
            param[index] = old
            numeric = (plus-minus)/2e-6
            max_error = max(max_error, float(abs(numeric-grad[index])))
            np.testing.assert_allclose(numeric, grad[index], rtol=2e-5, atol=2e-8)
    jac = hidden_jacobian(model, x[0])
    columns = []
    for coord in range(5):
        dx = np.zeros(5)
        dx[coord] = 1e-6
        plus = forward(model, (x[0]+dx)[None])[1][-1][0]
        minus = forward(model, (x[0]-dx)[None])[1][-1][0]
        columns.append((plus-minus)/2e-6)
    np.testing.assert_allclose(jac, np.stack(columns, axis=1), rtol=2e-5, atol=2e-8)
    linear = np.eye(12)
    for _ in range(16):
        linear = orthogonal(rng, 12) @ linear
    np.testing.assert_allclose(np.linalg.svd(linear, compute_uv=False), 1, atol=3e-14)
    # A single square ReLU layer zeros one Jacobian row per inactive unit.
    one_layer = initialize(16, 1, "orthogonal", 9)
    sample = rng.normal(size=16)
    pre = forward(one_layer, sample[None])[2][0][0]
    spec = spectrum(hidden_jacobian(one_layer, sample))
    assert spec["numerical_rank"] == int(np.sum(pre > 0))
    z = rng.normal(size=400000)
    activated = np.maximum(np.sqrt(2)*z, 0)
    np.testing.assert_allclose(np.mean(activated**2), 1, atol=.012)
    np.testing.assert_allclose(np.var(activated), 1-1/np.pi, atol=.012)
    c = .3
    a = np.maximum(z, 0)
    b = np.maximum(c*z + np.sqrt(1-c*c)*rng.normal(size=len(z)), 0)
    np.testing.assert_allclose(cosine(a, b), relu_correlation(c), atol=.006)
    # Varying width preserves pairwise inner products divided by width.
    physical = rng.normal(size=(7, 8))
    for width in (16, 32):
        embedded = physical @ embedding(width).T
        np.testing.assert_allclose(embedded @ embedded.T / width, physical @ physical.T / 8, atol=2e-14)
    # Every failed candidate remains visible and yields no selected rate.
    failed_runs = [run_training(8, 4, scheme, 0, 1e200, 5) for scheme in SCHEMES]
    assert all(not run["completed"] and run["failure"] and run["final_validation_loss"] is None for run in failed_runs)
    failed_summary = aggregate(failed_runs, (8,), (4,), (1e200,), (0,))
    assert all(row["selected_learning_rate"] is None for row in failed_summary)
    assert sum(row["selected_at_search_boundary"] is True for row in failed_summary) == 0
    # A fully inactive probe has an undefined cosine, reported as NA.
    dead = initialization_diagnostics(8, 12, "he", 14)
    assert dead["layers"][-1]["activation_pair_cosine"] is None
    assert format_optional(dead["layers"][-1]["activation_pair_cosine"]) == "NA"
    print(f"PASS: parameter/input finite differences (parameter max error {max_error:.2e})")
    print("PASS: orthogonal linear spectrum, ReLU gate rank, Gaussian moments/correlation, width-controlled input Gram")
    print("PASS: failed-run aggregation and undefined-cosine reporting")
    return {"parameter_finite_difference_max_absolute_error": max_error,
            "he_relu_second_moment_monte_carlo": float(np.mean(activated**2)),
            "he_relu_centered_variance_monte_carlo": float(np.var(activated)),
            "relu_correlation_at_point_three_monte_carlo": cosine(a, b),
            "relu_correlation_at_point_three_theory": relu_correlation(c)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="108 runs, 40 steps, three equally budgeted learning rates")
    parser.add_argument("--output", type=Path, help="write complete measurements and training traces as JSON")
    args = parser.parse_args()
    checks = self_checks()
    widths = (16, 32) if args.sweep else (16,)
    depths = (4, 12)
    seeds = (0, 1, 2) if args.sweep else (0, 1)
    rates = (.003, .01, .03) if args.sweep else (.01,)
    steps = 40 if args.sweep else 20
    diagnostics, runs = [], []
    for n in widths:
        for depth in depths:
            for scheme in SCHEMES:
                for seed in seeds:
                    diagnostics.append(initialization_diagnostics(n, depth, scheme, seed))
                    for lr in rates:
                        runs.append(run_training(n, depth, scheme, seed, lr, steps))
    summary = aggregate(runs, widths, depths, rates, seeds)
    print("\nFixed LR = 0.01; validation loss mean +/- sample std across seeds")
    print("width depth scheme     loss_mean loss_std failures; selected_LR_from_equal_budget")
    for row in summary:
        fixed = next(item for item in row["candidates"] if item["learning_rate"] == .01)
        mean, std = fixed["validation_loss_mean_completed"], fixed["validation_loss_std_completed"]
        mean_text = f"{mean:.6f}" if mean is not None else "NA"
        std_text = f"{std:.6f}" if std is not None else "NA"
        print(f"{row['width']:5d} {row['depth']:5d} {row['scheme']:10s} {mean_text} {std_text} {fixed['failures']}/{fixed['runs']}; {row['selected_learning_rate']}")
    print("\nInitialization, width 32 (or 16 in quick run), depth 12, seed 0")
    for scheme in SCHEMES:
        record = next(item for item in diagnostics if (item["width"], item["depth"], item["scheme"], item["seed"]) == (widths[-1], 12, scheme, 0))
        last = record["layers"][-1]
        spec = record["hidden_jacobian_at_probe_0"]
        print(f"{scheme:10s} hidden_E[x^2]={last['activation']['second_moment']:.6g} pair_cos={format_optional(last['activation_pair_cosine'])} J_rank={spec['numerical_rank']}/{widths[-1]} mean_s^2={spec['mean_squared_singular_value']:.6g} smax={spec['largest_singular_value']:.6g}")
    failures = sum(not run["completed"] for run in runs)
    print(f"\nCompleted {len(runs)-failures}/{len(runs)} runs; failures retained: {failures}.")
    if args.sweep:
        print(f"Best candidate at search boundary: {sum(row['selected_at_search_boundary'] is True for row in summary)}/{len(summary)} groups; extend the rate grid before claiming an optimum.")
    print("All rates, seed losses, per-layer moments, full singular spectra and update traces are retained with --output.")
    result = {
        "format_version": 1, "numpy_version": np.__version__, "mode": "sweep" if args.sweep else "quick",
        "configuration": {"widths": widths, "depths": depths, "seeds": seeds, "learning_rates": rates,
                          "steps": steps, "data_seed": DATA_SEED, "dtype": "float64", "optimizer": "full_batch_sgd",
                          "train_examples": 64, "validation_examples": 64, "probe_examples": 64,
                          "physical_input_dimension": 8, "weight_decay": 0, "momentum": 0,
                          "target": "sin(u[0]) + 0.3*u[1]*u[2]", "input_embedding": "P.T@P=(width/8)*I_8",
                          "loss": "mean((prediction-target)^2)/2", "fixed_comparison_learning_rate": .01},
        "measurement_definitions": {
            "moments": "Pooled over samples and coordinates; centered variance subtracts one pooled empirical mean.",
            "pair_cosine": "Uncentered cosine across feature coordinates for fixed physical observations 0 and 1; null if a vector is zero. A finite-width probe, not an ensemble expectation.",
            "wide_gaussian_reference": "Zero-bias Gaussian mean-field recursion starting from empirical input second moment and pair cosine; approximation for each finite network, also used as a wide-limit reference for orthogonal weights.",
            "covariance_effective_rank": "exp entropy of eigenvalues normalized by trace; covariance centers each neuron across 192 samples.",
            "gradient": "Backward propagation of the same seeded unit-norm vector through the hidden map for every sample; independent of readout/loss.",
            "jacobian": "Full width-by-width derivative of last hidden activation wrt embedded hidden input at probe sample 0. All singular values retained; rank uses width*float64_epsilon*smax.",
            "relative_weight_update": "L2 norm of this step's weight displacement / L2 norm of pre-step weights, per hidden layer; zero-start biases reported as absolute displacement.",
            "feature_drift": "RMS of current minus initialization activation on the same fixed probe batch, per layer.",
            "selection": "Choose lowest mean validation loss among learning rates with zero numeric failures. Population task/data fixed. Std uses ddof=1. Probe split is not used for selection.",
        },
        "checks": checks, "initialization": diagnostics, "runs": runs, "summary": summary,
    }
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
