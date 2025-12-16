Motto: "nos intelligentia sumus" -- vis-a-vis AI definition

## TODOs

  * fix Part 6:
     - implement semantic random walk: nothing back from the shortest paths?

  * fix Part 5:
     - also update `count` and `rank` for ER and TAXO nodes in the ERKG
     - fix the logic in `ctx.py: promote_er_nodes()`

  * refactor ERKG to its own module

  * Streamlit dashboard view of `rag.py`

  * filter the `strw:co_occurs_with` relations based on quantiles

  * fix `n3()` prefix resolution for `lex.py`, etc.

  * enable `Opik`

  * add edges for `strw:compound_elem_of`

  * build an HITL interface for accept/reject/override LLM suggestions on lex nodes w/o NER
     - `FastAPI` webapp, using `DSPy` suggestion
     - operate on the serialized files (offline) during Part 4
     - use autocompletion to guide selections based on domain context
     - actions:
         + NounChunk -> Lemma, or Synonym
	 + ParsedNoun -> Lemma, or Synonym     


  * formalize the description about the reranker process
     - https://www.lancedb.com/docs/reranking/custom-reranker/
     - use a "cross encoder" approach?
     - https://dubell.io/leveraging-bm25-and-vector-search-in-a-local-rag-application/

  * migrate from `Turtle` to `N3` format

  * `textrank`
     - rework with Polars in lieu of Pandas

  * integrate `retriv.SparseRetriever` for Okapi BM25 on text chunks
     - can we use `LanceDB` FTS w/o cloud?
     - https://lancedb.com/docs/search/full-text-search/


  * eval `GrandCypher` atop `NetworkX`
     - can this impl Text2Cypher efficiently?

  * build `RDFlib` plugin atop `NetworkX`
     - migrate `kglab.standards` for OWL/RDFS closure and SHACL capabilities
     - SHACL verify: NER nodes in ERKG should always have a `rdf:type` IRI class

  * use `deepeval` to score the GraphRAG responses
     - https://docs.google.com/presentation/d/1vDogMddS_T-oXFBdhogo653r8H2sGrSD/edit?slide=id.g78dc4dcc24882d5d_0#slide=id.g78dc4dcc24882d5d_0

  * gensim => ArrowSpace embeddings for entities
     - alteratively, use `sentence_transformer` in lieu of Gensim?

  * add a text => lemma embedding model
     - https://www.sbert.net/docs/sentence_transformer/usage/usage.html

  * replace RE by leveraging DSPy <https://arxiv.org/html/2502.09956v1>

  * entity linking - combine structured and unstructured sources using Sz results as a thesaurus


## Issues

  Q: how do we filter blank node constructs from SHACL rules?

  Q: how do we handle SKOS:Concept nodes with multiple lemmas? i.e., to produce a list from a SPARQL query?

  Q: why does `tracemalloc` cause `RDFlib` to block?



## RE prompts:

  * strw:member_of
        https://www.w3.org/TR/vocab-org/#org:memberOf

    - "which doctors are associated with which institutions?"
    - "which people are members of which organizations?"

  * strw:author_of
        http://purl.org/dc/elements/1.1/creator

    - "which experts commented on the study?"
    - "which people authored the study?"
  
  * strw:associated_with

    - "which food additives are associated with which conditions?"

  - "Organize these associations into a cause-effect style table (Behavior → Associated Condition) for clarity"
  - "Given the BEHAVIOR "eating a healthy diet", in the following text which conditions is that behavior associated?"


## Structured Data

DOI:
  - https://www.neurology.org/doi/10.1212/WNL.0000000000210286

Step 1:

Construct a structured dataset from D&B and OpenCorporates company
profiles, with duplicates as much as possible. Are there other
business directories which could be merged?

Step 2:
Leverage _disclosed relations_ among scholar profiles to link with
company profiles above -- even though spellings for their names,
affiliations, and locations may differ somewhat.

For example, Senzing ER could help resolve:

  * "Dong Wang" vs. "Daniel Wang"
  * "Jae H Kang" vs. "Jae-Hee Kang"
  * "Xiaohong Gu" vs. "Xiao Gu""

It's guaranteed that trying to resolve the author list here would
produce an utter mess of false identifications.

----------------------------------------------------------------------


## Semantics

    Person

Dr Heather Snyder
Dr Richard Oakley
Dr Yuhan Li
Yuhan Li
Heather M. Snyder


    People Groups

nurses
older adults
researchers


    City

Boston
Philadelphia

    Country

UK
US


    Organization

Alzheimer's Association
Women's Hospital
Alzheimer's Association International Conference
Alzheimer's Society
alz.org


    Food

bacon
ballgame hot dog
beans
bologna
hamburger
hot dog 
hotdogs
kielbasa
legumes
lentils
lima beans
nuts
peanut butter
peanuts
peas
pork chops
processed red meat
red meat
salami
sausage
soy milk
soy protein
steak
string beans
tofu
unprocessed red meat
walnuts


    Behaviors

meat consumption


    Condition

Alzheimer's
Alzheimer's disease
type 2 diabetes
heart disease
cognitive decline
dementia
cancer
diabetes


    Organ

brain


    Food Additives

preservatives
sodium
nitrites


    Publication

Brain Health


    Conference

AAIC

    Hospital


    University

Harvard T.H. Chan School


    Science

network medicine
public health
