# GraphGeeks.org talk 2024-08-14

How to construct _knowledge graphs_ from unstructured data sources.

See: <https://live.zoho.com/PBOB6fvr6c>


## Set up

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U pip wheel
python3 -m pip install -r requirements.txt 
```

## Run demo

The full demo app is in `demo.py`:

```bash
python3 demo.py
```

This will scrape text sources from particular URLs for articles about
the linkage between dementia and regularly eating procesed red meat,
then produces a knowledge graph using `NetworkX` and a vector database
of text chunck embeddings using `LanceDB`:

  * `kg.html` -- interactive graph visualization in `PyVis`
  * `kg.json` -- serialization of `NetworkX` graph
  * `data/lancedb` -- vector database tables


## Deep-dive

Multiple Jupyter notebooks illustrate important steps within this
workflow:

```bash
./venv/bin/jupyter-lab
```

  * Part 1: `construct.ipynb` -- detailed steps for KG construction from a lexical graph
  * Part 2: `chunk.ipynb` -- example of how to chunk text (YMMV)
  * Part 3: `vector.ipynb` -- query LanceDB table for chunked text embeddings (after running `demo.py`)
