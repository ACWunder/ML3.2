# ML-3

## How to run

### Setup

```bash
# 1) Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2) Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Make sure imports work from scripts
export PYTHONPATH="$PWD"
```
### Download, Clean and Split Data
```bash
# 1) Download raw data
python -m src.main download --config configs/preprocess.yaml

# 2) Clean & normalize text
python -m src.main clean    --config configs/preprocess.yaml

# 3) Split into train/val/test
python -m src.main split    --config configs/preprocess.yaml
```

### Running Shallow Baseline
```bash
# 1) Train LogisticRegression + BoW / TF–IDF
python -m src.models.train_shallow configs/train_shallow.yaml

# 2) Evaluate on test set
python -m src.models.evaluate_shallow configs/train_shallow.yaml
```

### Running DistilBERT

```bash
# 1) Virtual Environment aktivieren
source venv/bin/activate

# 2) PYTHONPATH setzen, damit `import src...` in Scripts funktioniert
export PYTHONPATH="$PWD"

# 4) Trainieren
python -m src.models.train_deep configs/deep_distilbert.yaml

# 5) Evaluieren
python -m src.models.evaluate_deep configs/deep_distilbert.yaml

```
### Running Bi-LSTM-Modell

```bash
# 1) Virtual Environment aktivieren
source venv/bin/activate

# 2) PYTHONPATH setzen, damit `import src...` in Scripts funktioniert
export PYTHONPATH="$PWD"

# 4) Trainieren
python -m src.models.train_deep configs/deep_bilstm.yaml

# 5) Evaluieren
python -m src.models.evaluate_deep configs/deep_bilstm.yaml
```