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

This demo scrapes text sources from articles about the linkage between
dementia and regularly eating processed red meat, then produces a graph
using `NetworkX`, a vector database of text chunk embeddings using
`LanceDB`, and an entity embedding model using `gensim.Word2Vec`,
where the results are:

  * `data/kg.json` -- serialization of `NetworkX` graph
  * `data/lancedb` -- vector database tables
  * `data/entity.w2v` -- entity embedding model
  * `kg.html` -- interactive graph visualization in `PyVis`

## Explore notebooks

A collection of Jupyter notebooks illustrate important steps
within this workflow:

```bash
./venv/bin/jupyter-lab
```

  * Part 1: `construct.ipynb` -- detailed KG construction using a lexical graph
  * Part 2: `chunk.ipynb` -- simple example of how to scrape and chunk text
  * Part 3: `vector.ipynb` -- query LanceDB table for text chunk embeddings (after running `demo.py`)
  * Part 4: `embed.ipynb` -- query the entity embedding model (after running `demo.py`)


## Generalized, unbundled process

**Objective:**
Construct a _knowledge graph_ (KG) using open source libraries where
deep learning models provide narrowly-focused _point solutions_ to
generate components for a graph: nodes, edges, properties.

These steps define a generalized process, where this tutorial picks up
at the _lexical graph_:

**Semantic overlay:**

  1. load any pre-defined controlled vocabularies directly into the KG

**Data graph:**

  1. load the structured data sources or updates into a data graph
  2. perform entity resolution (ER) on PII extracted from the data graph
  3. use ER results to generate a semantic overlay as a "backbone" for the KG

**Lexical graph:**

  1. parse the text chunks, using lemmatization to normalize token spans
  2. construct a lexical graph from parse trees, e.g., using a textgraph algorithm
  3. analyze named entity recognition (NER) to extract candidate entities from NP spans
  4. analyze relation extraction (RE) to extract relations between pairwise entities
  5. perform entity linking (EL) leveraging the ER results
  6. promote the extracted entities and relations up to the semantic overlay

This approach is in contrast to using a _large language model_ (LLM)
as a _one size fits all_ "black box" approach to generate the entire
graph automagically.
Black box approaches don't work well for KG practices in regulated environments, where audits, explanations, evidence, data provenance, etc., are required.

Better yet, review the intermediate results after each inference step to
collect human feedback for curating the KG components, e.g., using
[`Argilla`](https://github.com/argilla-io/argilla).

KGs used in mission-critical apps such as investigations generally rely
on updates, not a one-step construction process.
By producing a KG based on the steps above, updates can be handled more
effectively.
Downstream apps such as [_Graph RAG_](https://derwen.ai/s/hm7h)
for grounding the LLM results will also benefit from improved data quality.


## Component libraries

  * `spaCy`: <https://spacy.io/>
  * `GLiNER`: <https://github.com/urchade/GLiNER>
  * `GLiREL`: <https://github.com/jackboyla/GLiREL>
  * `OpenNRE`: <https://github.com/thunlp/OpenNRE>
  * `NetworkX`: <https://networkx.org/>
  * `PyVis`: <https://github.com/WestHealth/pyvis>
  * `LanceDB`: <https://github.com/lancedb/lancedb>
  * `gensim`: <https://github.com/piskvorky/gensim>
  * `pandas`: <https://pandas.pydata.org/>
  * `Pydantic`: <https://github.com/pydantic/pydantic>
  * `Pyinstrument`: <https://github.com/joerick/pyinstrument>

Note: you must use the `nre.sh` script to load OpenNRE pre-trained models before running the `opennre.ipynb` notebook.
