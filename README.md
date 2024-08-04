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
then produces a knowledge graph using `NetworkX`, a vector database of
text chunck embeddings using `LanceDB`, and an entity embedding models
using `gensim.Word2Vec`:

  * `kg.html` -- interactive graph visualization in `PyVis`
  * `data/kg.json` -- serialization of `NetworkX` graph
  * `data/lancedb` -- vector database tables
  * `data/entity.w2v` -- entity embedding model


## Deep-dive

Multiple Jupyter notebooks illustrate important steps within this
workflow:

```bash
./venv/bin/jupyter-lab
```

  * Part 1: `construct.ipynb` -- detailed steps for KG construction from a lexical graph
  * Part 2: `chunk.ipynb` -- example of how to chunk text (YMMV)
  * Part 3: `vector.ipynb` -- query LanceDB table for text chunk embeddings (after running `demo.py`)
  * Part 4: `embed.ipynb` -- query gensim.Word2Vec for entity embeddings (after running `demo.py`)
