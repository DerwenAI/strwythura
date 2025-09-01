## Where do we go next?

  * replace RE with DSPy <https://arxiv.org/html/2502.09956v1>
  * Streamlit dashboard view of `rag.py`
  * entity linking - combine structured and unstructured sources using Sz results as a thesaurus

  * back-out `pandas` replaced by `polars` in `textrank.py`
  * integrate `kglab.standards` for OWL/RDFS closure and SHACL capabilities
  * integrate with Kuzu for graph persistence?


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
