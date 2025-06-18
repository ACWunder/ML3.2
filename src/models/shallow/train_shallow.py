import os
import sys
import time
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.features.bow import build_vectorizer
from src.utils.config import load_config

def train_and_evaluate(cfg):
    # ──────────────────────────────────────────────────────────
    # ensure that output directories exist
    shallow_cfg = cfg['models']['shallow']
    for path_key in ['best_model_path', 'cv_results_path']:
        out_dir = os.path.dirname(shallow_cfg[path_key])
        os.makedirs(out_dir, exist_ok=True)
    # ──────────────────────────────────────────────────────────

    # 1) Daten einlesen
    train_df = pd.read_csv(cfg['split']['train_csv'])
    X_train = train_df['full_text']
    y_train = train_df['label']

    # 2) Pipeline bauen
    pipe = Pipeline([
        ('vec', build_vectorizer(cfg['features'])),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear')),
    ])

    # 3) Grid‐Parameter: ngram_ranges in Tuples umwandeln
    ngram_tuples = [tuple(r) for r in shallow_cfg['ngram_ranges']]
    param_grid = {
        'vec__ngram_range': ngram_tuples,
        'clf__C':           shallow_cfg['Cs'],
    }

    # 4) GridSearch starten & Zeit messen
    start = time.time()
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cfg['models']['cv_folds'],
        n_jobs=-1,
        verbose=2
    )
    gs.fit(X_train, y_train)
    duration = time.time() - start

    # 5) Best‐Model speichern
    joblib.dump(gs.best_estimator_, shallow_cfg['best_model_path'])

    # 6) CV‐Ergebnisse exportieren
    pd.DataFrame(gs.cv_results_) \
      .to_csv(shallow_cfg['cv_results_path'], index=False)

    # 7) Kurzes Logging
    print(f"✔ Finished training in {duration:.1f}s")
    print(f"✔ Best params: {gs.best_params_}")
    print(f"✔ Model saved to {shallow_cfg['best_model_path']}")

if __name__ == "__main__":
    cfg = load_config(sys.argv[1])
    train_and_evaluate(cfg)
