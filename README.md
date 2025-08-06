# Strwythura

Strwythura tutorial, based on a presentation for GraphGeeks.org on
2024-08-14

How to construct a _knowledge graph_ from unstructured data sources
using SOTA models for _named entity recognition_ (NER), and then
implement GraphRAG.

  * video: <https://youtu.be/B6_NfvQL-BE>
  * slides: <https://derwen.ai/s/2njz#1>

Caveat: this repo provides the source code and notebooks which
accompany an instructional tutorial; it is not intended as a package
library or maintained product.


## Set Up

```bash
poetry update
poetry run python3 -m spacy download en_core_web_md
```

## Run Demo

The full demo app is in `demo.py`:

```bash
poetry run python3 demo.py
```

This demo scrapes text sources from a set of URLs. The default set
includes articles about the linkage between dementia and regularly
eating processed red meat.

Then the demo iterates through multiple steps to produce results:

  1. Scrape each URL using `requests` and `BeautifulSoup`
  2. Split the text into _chunks_
  3. Build  _vector embeddings_ for each chunk, stored in `LanceDB`
  4. Parse each text chunk using `spaCy`, iterating per sentence
  5. Extract _entities_ from each sentence using `GLiNER`
  6. Build a _lexical graph_ from the parse trees in `NetworkX`
  7. Run a _textrank_ algorithm to rank important entities
  8. Build an embedding for each entity using `gensim.Word2Vec`
  9. Generate an interactive visualization using `PyVis`

There's also a step "5.1" which extracts _relations_ using `GLiREL`
though its results seem rather iffy so far.

Then a couple example queries get run to perform vector search for the
top-ranked text chunks and semantic expansion to enrich the set of
_anchor nodes_ in the graph.


Assets get serialized into these generated files:

  * `kg.html` -- interactive graph visualization in `PyVis`
  * `data/kg.json` -- serialization of `NetworkX` graph
  * `data/lancedb` -- vector database tables
  * `data/entity.w2v` -- entity embedding model

Note: this processing may take a few extra minutes the first time you
run it since `PyTorch` must download a large (~2GB) file.

BTW, if you'd had a graph previously constructed from more reliable
_structured data sources_, this demo could use a _semantic layer_ --
i.e., a "backbone" for the KG -- to organize the entities and
relations which get abstracted from from the lexical graph.


## Explore Notebooks

A collection of Jupyter notebooks illustrate important steps
within this workflow:

```bash
.venv/bin/jupyter-lab
```

  * Part 1: `construct.ipynb` -- detailed KG construction using a lexical graph
  * Part 2: `chunk.ipynb` -- simple example of how to scrape and chunk text
  * Part 3: `vector.ipynb` -- query LanceDB table for text chunk embeddings (after running `demo.py`)
  * Part 4: `embed.ipynb` -- query the entity embedding model (after running `demo.py`)


## Generalized, Unbundled Process

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

In contrast, many vendors suggest using a _large language model_ (LLM)
as a _one size fits all_ "black box" approach to generate an entire
graph automagically.

However, the business process of _resolution_ -- for both entities and
relations -- requires judgements. If the entities being resolved are
low-risk, low-effort in nature, then yeah knock yourself out. If the
entities represent _people_ or _organizations_, these have agency and
may take actions when misrepresent in applications which have
consequences.

Whenever judgements get delegated to _model-based_ approaches,
_generalization_ becomes a form of reasoning employed.  When the
technology within the model is based on _loss functions_, then
generalization becomes dominant -- regardless of any marketing claims
about "AI reasoning" made by tech firms.

Fortunately, decisions can be made _without models_, even in AI
applications. For more detailed discussion, see:

  * Part 1: Let's talk about "Today's AI" <https://www.linkedin.com/pulse/lets-talk-todays-ai-paco-nathan-co60c/>
  * Part 2: Let's talk about "Resolution" <https://www.linkedin.com/pulse/lets-talk-resolution-paco-nathan-ryjhc/>

Keep in mind that black box approaches don't work especially well
for KG practices in regulated environments, where audits,
explanations, evidence, data provenance, etc., are required.

Also keep in mind that KGs used in mission-critical apps, such as
investigations, generally require periodic data updates, so
construction isn't a one-step process. By producing a KG based on the
approach sketched above, updates can be handled more effectively.

Downstream usage such as [_GraphRAG_](https://derwen.ai/s/hm7h) for
grounding the LLM results also benefit from improved data quality in
the KG.


## Experimental - BAML

First, set up `BAML` <https://docs.boundaryml.com/guide/installation-language/python>

```bash
poetry run baml-cli generate
```

Second, download and install `Ollama` <https://ollama.com/> then pull
the Llama3 model:

```bash
ollama pull llama3:latest
```

Then run the GraphRAG example:

```bash
poetry run python3 rag.py
```


## Experimental - OpenNRE

The `OpenNRE` library also provides _relation extraction_ and there is
some experimental code which illustrates this.

  * `OpenNRE`: <https://github.com/thunlp/OpenNRE>

Use the `nre.sh` script to load OpenNRE pre-trained models before
running the `opennre.ipynb` notebook.

This may not work in many environments, depending on whether the
`OpenNRE` library is being maintained.


## FAQ

Q: "Have you tried this with `langextract` yet?"  
A: "I'll take `How an instructor knows a student ignored the README?` for $200"

Q: "Why aren't you using an LLM instead to build the graph?"  
A: "I promise to visit you in jail."


## License and Copyright

Source code, documentation, and examples have an
[MIT license](https://spdx.org/licenses/MIT.html)
which is succinct andsimplifies use in commercial applications.

All materials herein are Copyright © 2024-2025 Senzing, Inc.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=derwenai/strwythura&type=Date)](https://star-history.com/#derwenai/strwythura&Date)
