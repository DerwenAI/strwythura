# Strwythura

Strwythura tutorial, based on a presentation for <GraphGeeks.org> on
2024-08-14

How to construct a _knowledge graph_ from unstructured data sources
using SOTA models for _named entity recognition_ (NER), and then
implement GraphRAG.

  * video: <https://youtu.be/B6_NfvQL-BE>
  * slides: <https://derwen.ai/s/2njz#1>


## Set Up

```bash
poetry update
poetry run python3 -m spacy download en_core_web_md
```

## Usage

Caveat: this repo provides the source code and notebooks which
accompany an instructional tutorial; it is not intended as a packaged
library or maintained product.

That said, if you want to use this code to build an application it may
help to copy settings in `config.toml` into a custom configuration
file, then instantiate new `Strwythura` and `GraphRAG` objects using
it.


## Run Demo

The demo for constructing a knowledge graph, plus entity embeddings,
with nodes linked to chunks in a vector store is in the `demo.py`
script:

```bash
poetry run python3 demo.py
```

This scrapes text sources from a collection of URLs, given a set of
classes for extracted entities. The demo data includes articles about
the linkage between eating _processed red meat_ frequently and the
risks of _dementia_ later in life, based on long-term studies.

This demo iterates through multiple steps to produce the assets needed
for GraphRAG downstream:

  1. Scrape each URL using `requests` and `BeautifulSoup`
  2. Split the text into _chunks_
  3. Build  _vector embeddings_ for each chunk, stored in `LanceDB`
  4. Parse each text chunk using `spaCy`, iterating per sentence
  5. Extract _entities_ from each sentence using `GLiNER`
  6. Build a _lexical graph_ from the parse trees in `NetworkX`
  7. Run a _textrank_ algorithm to rank important entities
  8. Build an embedding model for the entities using `gensim.Word2Vec`
  9. Generate an interactive visualization using `PyVis`

There's also a step "5.1" which extracts _relations_ using `GLiREL`
though its results may be a bit sparse.

Note: processing may take a few extra minutes the first time it runs
since `PyTorch` must download a large (~2GB) file.

The assets get serialized into these files:

  * `data/lancedb` -- vector database tables in `LanceDB`
  * `data/kg.json` -- serialization of `NetworkX` graph
  * `data/sem.json` -- serialization of semantics for NER
  * `data/entity.w2v` -- entity embeddings in `Gensim`
  * `data/url_cache.sqlite` -- URL cache in `SQLite`
  * `kg.html` -- interactive graph visualization in `PyVis`

Note: if you had a graph previously constructed from more reliable
_structured data sources_, this demo could use a _semantic layer_ --
i.e., a "backbone" for the KG -- to organize the entities and
relations which get abstracted from from the lexical graph.


## About GraphRAG

A good downstream use case for exploring a newly constructed KG is
[_GraphRAG_](https://derwen.ai/s/hm7h), used for grounding the
responses by an LLM in a question/answer chat.

This implementation uses `BAML` <https://docs.boundaryml.com/home>
and leverages the KG using _semantic random walks_.

Note: the term "GraphRAG" means many different things ... see this
article for more details:
["Unbundling the Graph in GraphRAG"](https://www.oreilly.com/radar/unbundling-the-graph-in-graphrag/).

To set up, first download/install `Ollama` <https://ollama.com/>
and pull the Llama3 model:

```bash
ollama pull llama3:latest
```

Then run the `rag.py` script for an interactive GraphRAG example:

```bash
poetry run python3 rag.py
```


## Tutorial Notebooks

A collection of Jupyter notebooks illustrate important steps within
these workflows:

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

  1. Load any pre-defined controlled vocabularies directly into the KG.

**Data graph:**

  1. Load the structured data sources or updates into a data graph.
  2. Perform entity resolution (ER) on PII extracted from the data graph.
  3. Use ER results to generate a semantic overlay as a "backbone" for the KG.

**Lexical graph:**

  1. Parse the text chunks, using lemmatization to normalize token spans.
  2. Construct a lexical graph from parse trees, e.g., using a textgraph algorithm.
  3. Analyze named entity recognition (NER) to extract candidate entities from NP spans.
  4. Analyze relation extraction (RE) to extract relations between pairwise entities.
  5. Perform entity linking (EL) leveraging the ER results.
  6. Promote the extracted entities and relations up to the semantic overlay.

Of course many vendors suggest using a _large language model_ (LLM) as
a _one size fits all_ "black box" approach for extracting entities and
generating an entire graph automagically.

However, the business process of _resolution_ -- for both entities and
relations -- requires _judgements_. If the entities getting resolved
are low-risk, low-effort in nature, then yeah knock yourself out. If
the entities represent _people_ or _organizations_, these have agency
and may take actions when misrepresented in applications which have
consequences.

Whenever judgements get delegated to _model-based_ approaches,
_generalization_ becomes a form of reasoning employed.  When the
technology within the model is based on _loss functions_, then
generalization becomes dominant -- regardless of any marketing claims
about "AI reasoning" made by tech firms.

Fortunately, decisions can be made _without models_, even in AI
applications. Shock, horror!!! Plaease, say it isn't so!?! Brace
yourselves, using models is a thing, but not the only thing.  For more
detailed discussion, see:

  * Part 1: Let's talk about "Today's AI" <https://www.linkedin.com/pulse/lets-talk-todays-ai-paco-nathan-co60c/>
  * Part 2: Let's talk about "Resolution" <https://www.linkedin.com/pulse/lets-talk-resolution-paco-nathan-ryjhc/>

Also keep in mind that black box approaches don't work especially well
for regulated environments, where audits, explanations, evidence, data
provenance, etc., are required.

Moreover, KGs used in mission-critical apps, such as investigations,
generally require periodic data updates, so construction isn't a
one-step process. By producing a KG based on the approach sketched
above, updates can be handled more effectively.  Any downstream use
cases, such as AI applications, also benefit from improved quality of
semantics and representation.


## Experimental - OpenNRE

The `OpenNRE` library also provides _relation extraction_ and there is
some experimental code which illustrates this.

  * `OpenNRE`: <https://github.com/thunlp/OpenNRE>

Use the `nre.sh` script to load OpenNRE pre-trained models before
running the `opennre.ipynb` notebook.

This may not work in many environments, depending on whether the
`OpenNRE` library is being maintained.


## Developer Notes

After each `BAML` release update, some committer needs to regenerate
its Python client source:

```bash
poetry run baml-cli generate --from strwythura/baml_src
```


## FAQ

Q: "Have you tried this with `langextract` yet?"  
A: "I'll take `How an instructor knows a student ignored the README?` from the `FAFO` category, for $200"

Q: "What the hell is the name of this repo about?"  
A: "As you may have noticed, many open source projects by Derwen are named in a beautiful language called Gymraeg, which English speakers called 'Welsh', where this word [`strwythura`](https://translate.google.com/details?sl=cy&tl=en&text=strwythura&op=translate) translates as the verb **'structure'** in English."

Q: "Why aren't you using an LLM instead to build the graph?"  
A: "I promise to visit you in jail."


## License and Copyright

Source code, documentation, and examples have an
[MIT license](https://spdx.org/licenses/MIT.html)
which is succinct andsimplifies use in commercial applications.

All materials herein are Copyright © 2024-2025 Senzing, Inc.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=derwenai/strwythura&type=Date)](https://star-history.com/#derwenai/strwythura&Date)
