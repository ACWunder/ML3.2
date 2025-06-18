import yaml, json

def load_config(path):
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        else:
            return json.load(f)
