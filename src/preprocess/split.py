import pandas as pd, logging
from sklearn.model_selection import train_test_split
from src.utils.config import load_config

def main(cfg):
    df = pd.read_csv(cfg["split"]["input_csv"])
    logging.info(f"Splitting {len(df)} samples")
    ratios = cfg["split"]["ratios"]
    seed = cfg["split"]["random_seed"]

    # first split off test
    df_tmp, df_test = train_test_split(
        df, test_size=ratios[2], random_state=seed, stratify=df["label"]
    )
    # then split train vs val
    val_rel = ratios[1] / (ratios[0] + ratios[1])
    df_train, df_val = train_test_split(
        df_tmp, test_size=val_rel, random_state=seed, stratify=df_tmp["label"]
    )

    # write
    df_train.to_csv(cfg["split"]["train_csv"], index=False)
    df_val.to_csv(cfg["split"]["val_csv"],   index=False)
    df_test.to_csv(cfg["split"]["test_csv"], index=False)

    # log counts
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        counts = subset["label"].value_counts().to_dict()
        logging.info(f"{name} size {len(subset)}, class distribution {counts}")

if __name__ == "__main__":
    import argparse, logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    main(cfg)
