import subprocess, os, logging
from src.utils.config import load_config

def main(cfg):
    url = cfg["download"]["repo_url"]
    out = cfg["download"]["dest_dir"]
    logging.info(f"Cloning {url} → {out}")
    os.makedirs(out, exist_ok=True)
    # if already exists, skip or pull
    if os.listdir(out):
        subprocess.run(["git", "-C", out, "pull"], check=True)
    else:
        subprocess.run(["git", "clone", url, out], check=True)
    logging.info("Download complete.")

if __name__ == "__main__":
    from src.utils.config import load_config
    import argparse, logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    main(cfg)
