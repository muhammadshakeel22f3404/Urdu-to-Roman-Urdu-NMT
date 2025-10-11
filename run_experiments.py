import os
import glob
import argparse
import subprocess

def run(config_path: str):
    print(f"Running config: {config_path}")
    # Use absolute path for the config to avoid CWD issues
    subprocess.run(
        ["python", "-m", "nmt.train", "--config", config_path],
        check=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the configs directory next to this experiments/ folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_base = os.path.join(project_root, "configs")
    parser.add_argument("--base", type=str, default=default_base, help="Path to the configs directory.")
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    if not os.path.isdir(base):
        raise SystemExit(f"Configs directory not found: {base}")

    # Pick all exp*.json in the base directory
    cfg_paths = sorted(glob.glob(os.path.join(base, "exp*.json")))
    if not cfg_paths:
        raise SystemExit(f"No config files matching exp*.json found in {base}")

    print(f"Project root: {project_root}")
    print(f"Configs base: {base}")
    print("Configs to run:")
    for p in cfg_paths:
        print(" -", p)

    for cfg in cfg_paths:
        run(cfg)