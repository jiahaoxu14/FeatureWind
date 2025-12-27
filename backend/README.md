# Backend

Python core library, CLIs, datasets, docs, and the Flask API.

## Setup
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

Set `PYTHONPATH=src` (or install editable) when running commands.

## CLIs
- Generate tangent maps: `PYTHONPATH=src python cli/generate_tangent_map.py datasets/examples/iris/iris.csv tsne iris_tsne`
- Visualize: `PYTHONPATH=src python cli/createwind.py --tangent-map datasets/examples/iris/iris_tsne.tmap --top-k 5`

## API
```bash
cd backend
PYTHONPATH=src FLASK_APP=app.py flask run --port 5050
# or
PYTHONPATH=src python app.py
```
Uploads and run artifacts land in `var/`.
