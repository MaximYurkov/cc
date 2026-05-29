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
    cfg = json.loads(json.dumps(cfg))

    def actual_total(network):
        return (
            network["thalamus"]["E"] + network["thalamus"]["I"] +
            network["L4"]["E"] + network["L4"]["I"] +
            2 * network["L23"]["E"] +
            3 * network["L23"]["I"] +
            2 * network["L5"]["E"] +
            3 * (network["L5"]["I"] + network["L6"]["I"]) +
            network["L6"]["E"]
        )

    current_total = actual_total(cfg["network"])
    factor = target_total / current_total

    for layer in ["thalamus", "L4", "L23", "L5", "L6"]:
        cfg["network"][layer]["E"] = max(1, round(cfg["network"][layer]["E"] * factor))
        cfg["network"][layer]["I"] = max(1, round(cfg["network"][layer]["I"] * factor))

    diff = target_total - actual_total(cfg["network"])
    cfg["network"]["L6"]["E"] = max(1, cfg["network"]["L6"]["E"] + diff)

    return cfg


def count_total_neurons(network_cfg: dict) -> int:
    return (
        network_cfg["thalamus"]["E"] + network_cfg["thalamus"]["I"] +
        network_cfg["L4"]["E"] + network_cfg["L4"]["I"] +
        2 * network_cfg["L23"]["E"] +
        3 * network_cfg["L23"]["I"] +
        2 * network_cfg["L5"]["E"] +
        3 * (network_cfg["L5"]["I"] + network_cfg["L6"]["I"]) +
        network_cfg["L6"]["E"]
    )


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
    
    tmp_cfg_path = Path("network_realistic_10k_runtime.yaml")
    tmp_cfg_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8"
    )
    
    network_config = load_config(str(tmp_cfg_path))
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