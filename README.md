# Strwythura

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16934079.svg)](https://doi.org/10.5281/zenodo.16934079)
![Licence](https://img.shields.io/github/license/DerwenAI/strwythura)
![Repo size](https://img.shields.io/github/repo-size/DerwenAI/strwythura)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/DerwenAI/strwythura?style=plastic)


**Strwythura** library/tutorial, based on a presentation about GraphRAG for
[GraphGeeks](https://graphgeeks.org/) on 2024-08-14

<details>
  <summary><h2>Overview</h2></summary>

How to construct a _knowledge graph_ (KG) from unstructured data
sources using _state of the art_ (SOTA) models for _named entity
recognition_ (NER), then implement an enhanced _GraphRAG_ approach,
and curate semantics for optimizing AI app outcomes within a
specific domain.

  * videos: <https://youtu.be/B6_NfvQL-BE>, <https://senzing.com/gph-graph-rag-llm-knowledge-graphs/>
  * slides: <https://derwen.ai/s/2njz#1>

Motivation for this tutorial comes from the stark fact that the
term "GraphRAG" means many things, based on multiple conflicting
definitions. Several popular implementations reveal a relatively 
cursory understanding about either _natural language processing_ (NLP)
or graph algorithms, plus a _vendor bias_ toward their own query language.

See this article for more details and history:
["Unbundling the Graph in GraphRAG"](https://www.oreilly.com/radar/unbundling-the-graph-in-graphrag/).

Instead of delegating KG construction to a _large language model_
(LLM), this tutorial shows the use of sophisticated NLP pipelines
based on `spaCy`, `GLiNER`, _TextRank_, and related libraries.
Results are better/faster/cheaper, plus this provides more control
and oversight for _intentional arrangement_ of the KG. Then for
downstream usage in a question/answer chat bot, an enhanced GraphRAG
approach leverages graph algorithms (e.g., _semantic random walk_)
to optimize retrieval of text chunks which ultimately get presented
to an LLM for _summarization_ to produce responses.

For more detailed discussions, see:

  * enhanced GraphRAG: ["GraphRAG to enhance LLM-based apps"](https://derwen.ai/s/hm7h#3)
  * ontology pipeline: ["Intentional Arrangement"](https://jessicatalisman.substack.com/) by **Jessica Talisman**
  * `spaCy`: <https://spacy.io/>
  * `GLiNER`: <https://huggingface.co/urchade/gliner_base>
  * _TextRank_: <https://www.derwen.ai/docs/ptr/explain_algo/>

Some key issues regarding KG construction with LLMs which don't get
addressed much by the graph community and AI community in general:

  1. LLMs tend to mangle cross-domain semantics when used for building graphs; see _Mai2024_ referenced in the "GraphRAG to enhance LLM-based apps" talk above.
  2. You need to introduce a _semantic layer_ for representing the domain context, which follows more of a _neurosymbolic AI_ approach.
  3. Most all LLMs perform _question rewriting_ in ways which cannot be disabled, even when the `temperature` parameter is set to zero; this leads to relative degrees of "hallucinated questions" for which there are no clear workarounds.
  4. Any _model_ used for prediction introduces reasoning based on _generalization_, even more so when the model uses a _loss function_ for training; this tends to be the point where KG structure and semantics turn into crap; see the "Let's talk about ..." articles linked below.
  5. The approach outlined here is faster and less expensive, and produces better results than if you'd delegated KG construction to an LLM.

Of course, YMMV.

This approach leverages _neurosymbolic AI_ methods, combining
practices from:

  * _natural language processing_
  * _graph data science_
  * _entity resolution_
  * _ontology pipeline_
  * _context engineering_
  * _human-in-the-loop_

Overall, this illustrates a reference implementation for
_entity-resolved retrieval-augmented generation_ (ER-RAG).
</details>

---

![](./docs/assets/graph_vis.png)

---

## Usage in applications

This runs with Python 3.11, though the range of versions may be
extended soon.

To `pip` install from [PyPi](https://pypi.org/project/strwythura/):

```bash
python3 -m pip install strwathura
python3 -m spacy download en_core_web_md
```

Then to integrate this library within an application:

  1. Copy settings in `config.toml` into a custom configuration file.
  2. Subclass `DomainContext` to extend it for the use case.
  3. Define semantics in `domain.ttl` for the domain context.
  4. Run entity resolution to merge the structured datasets.
  5. Run `Ollama` and have already downloaded the Gemma3 LLM as described below.
  6. Instantiate new `DomainContext`, `Strwythura`, `VisHTML`, and `GraphRAG` objects or their subclassed extensions.
  7. ...
  8. Profit

Follow the patterns in the `build.py` and `errag.py` example scripts.

If you're working with documents in a language other than English,
well first that's absolutely fantastic, though next you need to:

  * Update model settings in the `config.toml` file.
  * Change the `spaCy` model downloaded here.
  * Also change the language tags used in `domain.ttl` as needed.


## Set up for demo or development

This library uses [`poetry`](https://python-poetry.org/docs/) for
package management, and first you need to install it. Then run:

```bash
poetry update
poetry run python3 -m spacy download en_core_web_md
```

<details>
  <summary><h2>Demo Part 1: Entity Resolution</h2></summary>

Run _entity resolution_ (ER) to produce entities and relations from
_structured data sources_, which tend to be more reliable than those
extracted from unstructured content.

What does this ER step buy us?  ER allows us to merge multiple
structured data sets, even without consistent _foreign keys_ being
available, producing an overlay of entities and relations among them
-- which is useful as a "backbone" for constructing a KG. Morever
when there are judgements being made from the KG about people or
organizations, ER provides accountability for the merge decisions.

This is especially important in public sector, healthcare, banking,
insurance -- i.e., in use cases where you might need to "send flowers"
when automated judgements go wrong.  For example, someone gets denied
a loan, has a medical insurance claim blocked, gets a tax audit, has
their voter registration voided, becomes the subject of an arrest
warrant, and so on.  In other words, people and organizations tend to
take legal actions when someone else causes them harm. You'll want an
audit trail of decisions based on evidence, when your software systems
are making these kinds of judgements.

For the domain context in this tutorial, say we have two hypothetical
datasets which provide business directory listings:

  * `sz_er/acme_biz.json` -- "ACME Business Directory"
  * `sz_er/corp_home.json` -- "Corporates Home UK"

Plus we have slices from datasets which provide listings about
researchers and scientific authors:

  * `sz_er/orcid.json` -- [ORCID](https://orcid.org/)
  * `sz_er/scopus.json` -- [Scopus](https://www.elsevier.com/products/scopus/data)

These four datasets can be merged using ER, with the results being a
domain-specific _thesaurus_ that generates graph elements: entities,
relations, properties. We'll blend this into our _semantic layer_ used
for organizing the KG later.


The following steps are optional, since these ER results have already
been pre-computed and provided in the `sz_er/export.json` file.
If you want to run [Senzing](https://senzing.com/docs/quickstart/)
to produce these ER results, use the following steps.

Senzing SDK runs in Python or Java, and can also be run as batch using
a container from DockerHub:

```bash
docker pull senzing/demo-senzing
```

Once this container is available, run:

```bash
docker run -it --rm --volume ./sz_er:/tmp/data senzing/demo-senzing
```

This brings up a Linux command line prompt `I have no name!` and the
local subdirectory `sz_er` will be mapped to the `/tmp/data' directory
Type the following commands for batch ER into the command line prompt.

First, set up the Senzing configuration for merging these datasets:

```bash
G2ConfigTool.py
```

Within the configuration tool, register the names of the data sources
being used:

```
addDataSource ACME_BIZ
addDataSource CORP_HOME
addDataSource ORCID
addDataSource SCOPUS
save
exit
```

Load each file and run ER on its data records:

```bash
G2Loader.py -f /tmp/data/acme_biz.json
G2Loader.py -f /tmp/data/corp_home.json
G2Loader.py -f /tmp/data/orcid.json
G2Loader.py -f /tmp/data/scopus.json
```

Export the ER results to the `sz_er/export.json` file, then exit the
container:

```bash
G2Export.py -F JSON -o /tmp/data/export.json
exit
```

This later gets parsed to produce the `data/thesaurus.ttl` file
(as RDF in "Turtle" format) during the next part of the demo to
augment the _semantic layer_.

</details>


<details>
  <summary><h2>Demo Part 2: Build Assets</h2></summary>

Given as input:

  * `domain.ttl` -- semantics for the domain context
  * `sz_er/export.json` -- a domain-specific thesaurus based on ER
  * a list of structured datasets used in ER
  * a list of URLs from which to scrape content

The `domain.ttl` file provides a basis for iterating with an _ontology
pipeline_ process, to represent the semantics for the given domain.
It specifies metadata in terms of _vocabulary_, _taxonomy_, and
_thesaurus_ -- to use in representing the core entities and relations
in the KG.

The `curate.py` script described below then will introduce the
_human-in-the-loop_ part of this process, where you can review
entities extracted from documents. Based on this analysis, decide
where to refine the domain context to be able to _extract_,
_classify_, and _connect_ more of what gets extracted from
_unstructured data sources_ and linked into the KG.  Overall, this
process distills elements of the _lexical graph_, linking them with
elements from the _data graph_, to produce a more abstracted (i.e.,
less noisy) _semantic layer_ as the resulting KG.

Meanwhle, let's get started. The `build.py` script scrapes text
sources and constructs a _knowledge graph_ plus _entity embeddings_,
with nodes linked to chunks in a _vector store_:

```bash
poetry run python3 build.py
```

Demo data used in this case includes articles about the linkage
between eating _processed red meat_ frequently and the risks of
_dementia_ later in life, based on long-term studies.

The approach in this tutorial iterates through multiple steps to
produce the assets needed for GraphRAG downstream:

  1. Scrape each URL using `requests` and `BeautifulSoup`
  2. Split the text into _chunks_
  3. Build _vector embeddings_ for each chunk, in `LanceDB`
  4. Parse each text chunk using `spaCy`, iterating per sentence
  5. Extract _entities_ from each sentence using `GLiNER`
  6. Build a _lexical graph_ from the parse trees in `NetworkX`
  7. Run a _textrank_ algorithm to rank important entities
  8. Build an embedding model for entities using `gensim.Word2Vec`
  9. Generate an interactive visualization using `PyVis`

Note: processing may take a few extra minutes the first time it runs
since `PyTorch` must download a large (~2GB) file.

The assets get serialized into these files:

  * `data/lancedb` -- vector database tables in `LanceDB`
  * `data/kg.json` -- serialization of `NetworkX` graph
  * `data/sem.csv` -- entity semantics from `curate.py`
  * `data/entity.w2v` -- entity embeddings in `Gensim`
  * `data/url_cache.sqlite` -- URL cache in `SQLite`
  * `kg.html` -- interactive graph visualization in `PyVis`
</details>

<details>
  <summary><h2>Demo Part 3: Enhanced GraphRAG chat bot</h2></summary>

A good downstream use case for exploring a newly constructed KG is
GraphRAG, used for grounding the responses by an LLM in a
question/answer chat.

This implementation uses `BAML` <https://docs.boundaryml.com/home>
and leverages the KG using _semantic random walks_.

To set up, first download/install `Ollama` <https://ollama.com/>
and pull the Gemma3 model <https://huggingface.co/google/gemma-3-12b-it>

```bash
ollama pull gemma3:12b
```

Then run the `errag.py` script for an interactive GraphRAG example:

```bash
poetry run python3 errag.py
```
</details>

<details>
  <summary><h2>Demo Part 4: Curating an Ontology Pipeline</h2></summary>

This code uses a _semantic layer_ -- in other words, a "backbone" for
the KG -- to organize the entities and relations which get abstracted
from the lexical graph.

For now, run the `curate.py` script to generate a view of the ranked
NER results, serialized as the `data/sem.csv` file.  This can be
viewed in a spreadsheet to understand how to iterate on the semantic
definitions for more effective graph organization in the domain of the
scraped documents.

```bash
poetry run python3 curate.py
```
</details>

---

![](./docs/assets/q_a.1.png)

---

<details>
  <summary><h2>Unbundling GraphRAG</h2></summary>

**Objective:**

Construct a _knowledge graph_ (KG) using open source libraries where
deep learning models provide narrowly-focused _point solutions_ to
generate components for a graph: nodes, edges, properties.

These steps define a generalized process, where this tutorial picks up
at the _lexical graph_, without the _entity linking_ (EL) part yet:

**Semantic layer:**

  1. Load any semantics for domain context from pre-defined controlled vocabularies, taxonomies, thesauri, ontologies, etc., directly into the KG.

**Data graph:**

  1. Load the structured data sources or updates into a data graph.
  2. Perform _entity resolution_ (ER) on PII extracted from the data graph.
  3. Blend the ER results into the semantic layer as a "backbone" for structuring the KG.

**Lexical graph:**

  1. Parse the text chunks, using lemmatization to normalize token spans.
  2. Construct a lexical graph from parse trees, e.g., using a textgraph algorithm.
  3. Analyze _named entity recognition_ (NER) to extract candidate entities from noun phrase spans.
  4. Analyze _relation extraction_ (RE) to extract relations between pairwise entities.
  5. Perform _entity linking_ (EL) leveraging the ER results.
  6. Promote the extracted entities and relations up to the semantic layer.

Of course many vendors suggest using a _large language model_ (LLM) as
a _one-size-fits-all_ (OSFA) "black box" approach for extracting
entities and generating an entire graph **automagically**.

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
applications. Shock, horror!!! Please, say it isn't so!?! Brace
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
</details>


<details>
  <summary><h2>FAQ</h2></summary>

<dl>
<dt>Q:</dt><dd>"Have you tried this with <code>langextract</code> yet?"</dd>
<dt>A:</dt><dd>"I'll take <code>How does an instructor know a student ignored the README?</code> from the <a href="https://en.wiktionary.org/wiki/fuck_around_and_find_out" target="_blank"><em>What is FAFO?</em></a> category, for $200 ... but yes of course, it's an interesting package, which builds on other interesting work used here. Except that key parts of it miss the point entirely, in ways that only a hyperscaler could possibly fuck up so badly."</dd>
</dl>

<dl>
<dt>Q:</dt><dd>"What the hell is the name of this repo about?"</dd>
<dt>A:</dt><dd>"As you may have noticed, many open source projects published in this GitHub organization are named in a beautiful language Gymraeg, which English speakers call 'Welsh', where this word <a href="https://translate.google.com/details?sl=cy&tl=en&text=strwythura&op=translate" target="_blank"><code>strwythura</code></a> translates as the verb <strong>structure</strong> in English."</dd>
</dl>

<dl>
<dt>Q:</dt><dd>"Why aren't you using an LLM to build the graph instead?"</dd>
<dt>A:</dt><dd>"I promise to visit you in jail."</dd>
</dl>

<dl>
<dt>Q:</dt><dd>"Um, yeah, like, didn't Karpathy say to use <em>vibe coding</em>, or something? #justsayin"
<dt>A:</dt><dd>"<a href="https://effinbirds.com/" target="_blank">Piss the eff off</a> tech bro. Srsly, like yesterday -- you're embarrassing our entire industry with your overly exuberant ignorance."</dd>
</dl>

</details>


<details>
  <summary>Developer Notes</summary>

After each `BAML` release update, some committer needs to regenerate
its Python client source:

```bash
poetry run baml-cli generate --from strwythura/baml_src
```
</details>


<details>
  <summary>Experimental: Relation Extraction evaluation</summary>

Current Python libraries for _relation extraction_ (RE) are
probably best characterized as "experimental research projects".

Their tokenization approaches tend to make the mistake of "throwing
the baby out with the bath water" by not leveraging other available
information, e.g., what we have in the _textgraph_ representation of
the parsed documents. Also, they tend to ignore the semantic
constraints of the domain context, while computationally boiling
the ocean.

RE libraries which have been evaluated:

  * `GLiREL`: <https://github.com/jackboyla/GLiREL>
  * `ReLIK`: <https://github.com/SapienzaNLP/relik>
  * `OpenNRE`: <https://github.com/thunlp/OpenNRE>
  * `mREBEL`: <https://github.com/Babelscape/rebel>

This project had used `GLiREL` although its results were quite sparse.
RE will be replaced by `BAML` or `DSPy` workflows in the near future.

There is some experimental code which illustrates `OpenNRE` evaluation.
Use the `archive/nre.sh` script to load OpenNRE pre-trained models
before running the `archive/opennre.ipynb` notebook.

This may not work in many environments, depending on how well the
`OpenNRE` library is being maintained.
</details>


<details>
  <summary>Experimental: Tutorial notebooks</summary>

<p>
A collection of Jupyter notebooks were used to prototype code. These
help illustrate important intermediate steps within these workflows:
</p>

```bash
.venv/bin/jupyter-lab
```

<ul>
<li>`archive/construct.ipynb` -- detailed KG construction using a lexical graph</li>
<li>`archive/chunk.ipynb` -- simple example of how to scrape and chunk text</li>
<li>`archive/vector.ipynb` -- query LanceDB table for text chunk embeddings (after running `build.py`)</li>
<li>`archive/embed.ipynb` -- query the entity embedding model (after running `build.py`)</li>
</ul>

<p>
These are now archived, though kept available for study.
</p>
</details>


<details>
  <summary>License and Copyright</summary>

Source code for **Strwythura** plus its logo, documentation, and examples
have an [MIT license](https://spdx.org/licenses/MIT.html) which is
succinct and simplifies use in commercial applications.

All materials herein are Copyright © 2024-2025 Senzing, Inc.
</details>


<details>
  <summary>Kudos and Attribution</summary>

Please use the following BibTeX entry for citing **Strwythura** if you use
it in your research or software.
Citations are helpful for the continued development and maintenance of
this library.

```bibtex
@software{strwythura,
  author = {Paco Nathan},
  title = {{Strwythura: construct a knowledge graph from unstructured data sources, organized by results from entity resolution, implementing an enhanced GraphRAG approach, and also implementing an ontology pipeline plus context engineering for optimizing AI application outcomes within a specific domain}},
  year = 2024,
  publisher = {Senzing},
  doi = {10.5281/zenodo.16934079},
  url = {https://github.com/DerwenAI/strwythura}
}
```

Kudos to 
[@louisguitton](https://github.com/louisguitton), 
[@cj2001](https://github.com/cj2001),
[@prrao87](https://github.com/prrao87), 
[@hellovai](https://github.com/hellovai),
[@docktermj](https://github.com/docktermj), 
[@jbutcher21](https://github.com/jbutcher21),  
and the kind folks at [GraphGeeks](https://graphgeeks.org/) for their support.
</details>


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=derwenai/strwythura&type=Date)](https://star-history.com/#derwenai/strwythura&Date)
