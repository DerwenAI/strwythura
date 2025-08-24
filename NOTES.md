
## Where do we go next?

  * add a SKOS-based semantic layer, to leverage during graph algos in RAG
    - TODO: add SKOS:Concept definitions as "chunks" in `LanceDB`
    - TODO: add concepts into the `NetworkX` graph, linking entities

  * Streamlit dashboard view of `rag.py`

  * Entity linking - combine structured and unstructured sources


## Code TODOs
  - replace RE with BAML or DSPy <https://arxiv.org/html/2502.09956v1>

  * back-out `pandas` replaced by `polars`:
    - `textrank.py`

  * integrate `kglab.standards` for OWL/RDFS closure and SHACL capabilities

  * integrate with Kuzu for graph persistence?


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
