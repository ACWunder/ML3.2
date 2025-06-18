import pandas as pd, re, logging
from src.utils.config import load_config

def clean_text(s, regex):
    s = s.lower()
    return re.sub(regex, " ", s)

def main(cfg):
    df = pd.read_csv(cfg["clean"]["input_csv"])
    logging.info(f"Loaded {len(df)} records")
    # combine
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    # lowercase & remove
    df["full_text"] = df["full_text"].apply(
        lambda t: clean_text(t, cfg["clean"]["remove_regex"])
    )
    # map labels
    df["label"] = df["label"].map(cfg["clean"]["label_map"])
    out = cfg["clean"]["output_csv"]
    df[["full_text","label"]].to_csv(out, index=False)
    logging.info(f"Wrote cleaned data to {out}")

if __name__ == "__main__":
    import argparse, logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    main(cfg)
