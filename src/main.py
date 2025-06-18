import argparse
from src.preprocess import download, clean, split
from src.utils.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["download","clean","split"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    if args.stage == "download":
        download.main(load_config(args.config))
    elif args.stage == "clean":
        clean.main(load_config(args.config))
    elif args.stage == "split":
        split.main(load_config(args.config))
