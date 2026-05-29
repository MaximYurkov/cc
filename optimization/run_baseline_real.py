import argparse
import json
import time
from pathlib import Path

import yaml
import numpy as np

from core.simulator import run_simulation, load_config, SimulationConfig
from core.parameters import get_parameter_space
from core.objective import ObjectiveFunction
from core.visualization import SimulationVisualizer


def scale_to_target_neurons(cfg: dict, target_total: int = 10000) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy через json
    layers = ["thalamus", "L4", "L23", "L5", "L6"]

    current_total = sum(cfg["network"][l]["E"] + cfg["network"][l]["I"] for l in layers)
    factor = target_total / current_total

    for l in layers:
        cfg["network"][l]["E"] = round(cfg["network"][l]["E"] * factor)
        cfg["network"][l]["I"] = round(cfg["network"][l]["I"] * factor)

    new_total = sum(cfg["network"][l]["E"] + cfg["network"][l]["I"] for l in layers)
    diff = target_total - new_total
    cfg["network"]["L6"]["E"] += diff

    return cfg


def count_total_neurons(network_cfg: dict) -> int:
    layers = ["thalamus", "L4", "L23", "L5", "L6"]
    return sum(network_cfg[l]["E"] + network_cfg[l]["I"] for l in layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="network_realistic.yaml")
    parser.add_argument("--target-neurons", type=int, default=10000)
    parser.add_argument("--conn-prob", type=float, default=0.05)
    parser.add_argument("--tstop", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-dir", default="results")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cfg = scale_to_target_neurons(cfg, args.target_neurons)
    cfg["simulation"]["tstop"] = args.tstop
    cfg["simulation"]["dt"] = args.dt

    network_config = load_config(cfg)
    default_params = get_parameter_space().get_default()

    sim_config = SimulationConfig(
        use_gpu=False,
        num_threads=args.threads,
        cache_efficient=True,
        verbose=True
    )

    objective = ObjectiveFunction()

    run_name = f"baseline_real_{args.target_neurons}_cp_{str(args.conn_prob).replace('.', '_')}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    spike_data = run_simulation(
        default_params,
        network_config=network_config,
        sim_config=sim_config,
        conn_prob=args.conn_prob,
        seed=42,
        return_traces=False
    )
    elapsed = time.time() - start

    fitness = objective(spike_data)

    visualizer = SimulationVisualizer(save_dir, show=False)
    visualizer.plot_all(
        spike_data,
        prefix="baseline_",
        params_info=default_params,
        trace_data=None
    )

    result = {
        "config_base": args.config,
        "target_neurons": args.target_neurons,
        "actual_total_neurons": count_total_neurons(cfg["network"]),
        "conn_prob": args.conn_prob,
        "tstop": args.tstop,
        "dt": args.dt,
        "threads": args.threads,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "fitness": fitness,
        "save_dir": str(save_dir),
    }

    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("DONE")
    print("target_neurons =", args.target_neurons)
    print("actual_total_neurons =", result["actual_total_neurons"])
    print("conn_prob =", args.conn_prob)
    print("time_min =", round(result["elapsed_min"], 2))
    print("fitness =", fitness)
    print("saved to =", save_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()