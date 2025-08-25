
## Where do we go next?

  * add a SKOS-based semantic layer, to leverage during graph algos in RAG
    - TODO: add SKOS:Concept definitions as "chunks" in `LanceDB`
    - TODO: add concepts into the `NetworkX` graph, linking entities

  * Streamlit dashboard view of `rag.py`

  * Entity linking - combine structured and unstructured sources


## Structured Data

DOI:
  - https://www.neurology.org/doi/10.1212/WNL.0000000000210286

Step 1:

Construct a structured dataset from D&B and OpenCorporates company
profiles, with duplicates as much as possible. Are there other
business directories which could be merged?

D&B:
  - https://www.dnb.com/business-directory/company-profiles.american_academy_of_neurology.6bcbda77a7ce131dc0a3cf8cd1dcc509.html
  - https://www.dnb.com/business-directory/company-profiles.alzheimer_association.a356de6bbc6e48a378af50fa9ad3cf47.html
  - https://www.dnb.com/business-directory/company-profiles.alzheimers_association.871c8b05f9111fc253322e7b95d5af34.html
  - https://www.dnb.com/business-directory/company-profiles.alzheimers_society.7c2be519bc8cdda1a8bbdf91d1fb4b43.html
  - https://www.dnb.com/business-directory/company-profiles.harvard_medical_school.49c6b7f0cc27e39c780f6e27c8493c5c.html
  - https://www.dnb.com/business-directory/company-profiles.harvard_th_chan_school_of_public_health.160836775c9f888ec4ecd4b0f1378eb6.html
  - https://www.dnb.com/business-directory/company-profiles.the_brigham_and_womens_hospital_inc.9dfee810ea9418d160114a351d939a78.html
  - https://www.dnb.com/business-directory/company-profiles.the_brigham_and_womens_hospital_inc.e5e165d066fc8afaa92237df7df482d8.html
  - https://www.dnb.com/business-directory/company-profiles.mass_general_brigham_incorporated.c9a6480d962bd5c041787fde9d41bbe9.html
  - https://www.dnb.com/business-directory/company-profiles.national_institutes_of_health.043e4401a91e8b87f1fe354823d7f0fb.html

OpenCorporates:
  - https://opencorporates.com/companies/us_co/20251053702
  - https://opencorporates.com/companies/us_mn/e8ee97d5-8fd4-e011-a886-001ec94ffe7f
  - https://opencorporates.com/companies/us_md/D06554489
  - https://opencorporates.com/companies/us_md/F16412603


Step 2:
Leverage _disclosed relations_ among scholar profiles to link with
company profiles above -- even though spellings for their names,
affiliations, and locations may differ somewhat.

Scopus:
  - Yuhan Li, https://www.scopus.com/authid/detail.uri?authorId=59526840200
  - Heather Snyder, https://www.scopus.com/authid/detail.uri?authorId=7102670847
  - Daniel Wang, https://www.scopus.com/authid/detail.uri?authorId=56351539800
  - Richard Oakley, https://www.scopus.com/authid/detail.uri?authorId=9940916600
  - Heather Eliassen, https://www.scopus.com/authid/detail.uri?authorId=8067876900
  - Walter Willet, https://www.scopus.com/authid/detail.uri?authorId=57205132302
  - Jae-Hee Kang, https://www.scopus.com/authid/detail.uri?authorId=57225852260
  - Meir Stampfer, https://www.scopus.com/authid/detail.uri?authorId=55541278700
  - Xiaohong Gu, https://www.scopus.com/authid/detail.uri?authorId=57193551845
  - Molin Wang, https://www.scopus.com/authid/detail.uri?authorId=12774321000
  - Yaning Li, https://www.scopus.com/authid/detail.uri?authorId=57200572069

ORCID:
  - Heather Eliassen, https://orcid.org/0000-0002-3961-6609
  - Yuxi Liu, https://orcid.org/0000-0003-2484-151X
  - Molin Wang, https://orcid.org/0000-0003-1951-8961
  - Meir Stampfer, https://orcid.org/0000-0001-8865-935X
  - Yanping Li, https://orcid.org/0000-0002-0412-2748
  - Danyue Dong, https://orcid.org/0009-0003-3256-2876
  - Jae H Kang, https://orcid.org/0000-0003-4812-0557
  - Walter Willett, https://orcid.org/0000-0003-1458-7597
  - Dong Wang, https://orcid.org/0000-0002-0897-3048

For example, Senzing ER could help resolve:

  * "Dong Wang" vs. "Daniel Wang"
  * "Jae H Kang" vs. "Jae-Hee Kang"
  * "Xiaohong Gu" vs. "Xiao Gu""

It's guaranteed that trying to resolve the author list here would
produce an utter mess of false identifications.


ResearchGate:
  - Yanping Li, https://www.researchgate.net/profile/Yanping-Li-5
  - Walter Willett, https://www.researchgate.net/profile/Walter-Willett

DLBP:
  - Danyue Dong, https://dblp.org/pid/249/1428.html

Wikimedia:
  - https://en.wikipedia.org/wiki/A._Heather_Eliassen



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
