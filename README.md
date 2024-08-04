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
the linkage between dementia and regularly eating processed red meat,
then produces a knowledge graph using `NetworkX`, a vector database of
text chunk embeddings using `LanceDB`, and an entity embedding models
using `gensim.Word2Vec`:

  * `kg.html` -- interactive graph visualization in `PyVis`
  * `data/kg.json` -- serialization of `NetworkX` graph
  * `data/lancedb` -- vector database tables
  * `data/entity.w2v` -- entity embedding model


## Explore notebooks

Multiple Jupyter notebooks illustrate important steps within this
workflow:

```bash
./venv/bin/jupyter-lab
```

  * Part 1: `construct.ipynb` -- detailed steps for KG construction using a lexical graph
  * Part 2: `chunk.ipynb` -- a simple example of how to scrape and chunk text
  * Part 3: `vector.ipynb` -- query LanceDB table for text chunk embeddings (after running `demo.py`)
  * Part 4: `embed.ipynb` -- query gensim.Word2Vec model for entity embeddings (after running `demo.py`)


## Generalized process

**Objective:**
Construct a _knowledge graph_ (KG) using open source libraries with
deep learning models as _point solutions_ to generate components of
the graph.

The steps in a generalized process are the following, where this tutorial picks up at step 5:

  1. load any pre-defined _controlled vocabulary_ (layer 3)
  2. load the _structured data_ sources into a _data graph_ (layer 1)
  3. perform _entity resolution_ (ER) on PII extracted from the _data graph_
  4. use ER results to generate a _semantic overlay_ as a "backbone" for the KG (layer 3)
  5. scrape URLs to obtain the _unstructured data_ sources
  6. split the text into chunks and load into a vector database
  7. parse the text chunks, using _lemmatization_ to normalize token spans
  8. construct a _lexical graph_ from parse trees using the _textrank_ algorithm (layer 2)
  9. analyze _named entity recognition_ (NER) to extract candidate entities from spans
  10. analyze _relation extraction_ (RE) to extract relations between pairwise entities
  11. perform _entity linking_ (EL) leveraging the ER results
  12. promote the extracted entities and relations into the KG (layer 3)


Better yet, review the intermediate results at steps [4, 9, 10, 11] to
collect human feedback for curating the KG, for example by using
[`Argilla`](https://github.com/argilla-io/argilla).

Once you have produced a KG following this process, updates can be handled more robustly, and downstream apps such as
[_Graph-enhanced RAG_](https://discord.gg/N9A83zuhZu)
for grounding LLM results can work with better data quality.

Note: this approach is in contrast to using a _large language model_ (LLM) as a _one size fits all_ "black box" approach to generate the entire graph
automagically.
Black box approaches don't work well for KG practices in regulated environments, where audits, explanations, evidence, data provenance, etc., are required.


## Component libraries

  * `spaCy`: <https://spacy.io/>
  * `GLiNER`: <https://github.com/urchade/GLiNER>
  * `GLiREL`: <https://github.com/jackboyla/GLiREL>
  * `ReLIK`: <https://github.com/SapienzaNLP/relik>
  * `NetworkX`: <https://networkx.org/>
  * `PyVis`: <https://github.com/WestHealth/pyvis>
  * `LanceDB`: <https://github.com/lancedb/lancedb>
  * `gensim`: <https://github.com/piskvorky/gensim>
  * `pandas`: <https://pandas.pydata.org/>
  * `Pydantic`: <https://github.com/pydantic/pydantic>
  * `Pyinstrument`: <https://github.com/joerick/pyinstrument>
