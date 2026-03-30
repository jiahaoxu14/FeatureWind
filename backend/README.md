# Backend

Python core library, tangent-map generation, datasets, docs, and the Flask API.

## Setup
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

Set `PYTHONPATH=src` (or install editable) when running commands.

## Tangent Maps
- Generate tangent maps: `./.venv/bin/python src/generate_tangent_map.py datasets/examples/iris/iris.csv tsne iris_tsne`

## API
```bash
cd backend
PYTHONPATH=src FLASK_APP=app.py flask run --port 5050
# or
PYTHONPATH=src python app.py
```
Uploads and run artifacts land in `var/`.
