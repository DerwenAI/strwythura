#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Parse the Senzing entity resolution results exported as JSON.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import pathlib
import sys
import typing

from icecream import ic

SZ_PREFIX: str = "sz_entity_"
LANG_EN: str = "en"


if __name__ == "__main__":
    org_map: typing.Dict[ str, str ] = {}
    parent: typing.Dict[ str, str ] = {}
    rdf_list: typing.List[ str ] = []

    # load the data records
    data_records: typing.Dict[ str, dict ] = {}

    datasets: typing.List[ str ] = [
        "acme_biz.json",
        "corp_home.json",
        "orcid.json",
        "scopus.json",
    ]

    for filename in datasets:
        data_path: pathlib.Path = pathlib.Path(filename)

        with open(data_path, encoding = "utf-8") as fp:
            for line in fp:
                rec: dict = json.loads(line)
                record_id: str = rec["DATA_SOURCE"].replace(" ", "_").lower() + "_" + rec["RECORD_ID"]
                data_records[record_id] = rec

    # parse the JSON export
    export_path: pathlib.Path = pathlib.Path("export.json")

    with open(export_path, encoding = "utf-8") as fp:
        for line in fp:
            data: str = json.loads(line)

            entity_id: str = SZ_PREFIX + str(data["RESOLVED_ENTITY"]["ENTITY_ID"])
            ent_descrip: str = ""
            ent_type: str = ""

            rec_list: typing.List[ dict ] = []
            rel_list: typing.List[ dict ] = []

            for rec in data["RESOLVED_ENTITY"]["RECORDS"]:
                ent_descrip = rec["ENTITY_DESC"]

                record_id: str = rec["RECORD_ID"]
                data_source: str = rec["DATA_SOURCE"].replace(" ", "_").lower()
                rec_iri: str = f"{data_source}_{record_id}"

                parent[rec_iri] = entity_id

                pred_iri: str = "skos:exactMatch"

                rec_list.append({
                    "pred": pred_iri,
                    "obj": rec_iri,
                    "skos:prefLabel": rec["ENTITY_DESC"],
                })

            for rel in data["RELATED_ENTITIES"]:
                match_key: str = rel["MATCH_KEY"]
                match_level: int = rel["MATCH_LEVEL"]
                match_code: str = rel["MATCH_LEVEL_CODE"]

                why: str = f"{match_key} {match_level}"
                pred_iri: str = "skos:related"

                if match_code == "POSSIBLY_SAME":
                    pred_iri: str = "skos:closeMatch"

                rel_list.append({
                    "pred": pred_iri,
                    "obj": SZ_PREFIX + str(rel["ENTITY_ID"]),
                    "skos:definition": why,
                })

            ent_node: dict = {
                "iri": entity_id,
                "skos:prefLabel": ent_descrip,
            }

            ic(ent_node)

            rdf_frag: str = f"{entity_id} skos:prefLabel \"{ent_descrip}\"@\"{LANG_EN}\" "

            for rec_node in rec_list:
                dat_rec: dict = data_records[rec_node["obj"]]
                ent_type = dat_rec["RECORD_TYPE"]
                rdf_frag += f';\n  {rec_node["pred"]} {rec_node["obj"]} '
                #print("record", rec_node, dat_rec)

                if ent_type == "ORGANIZATION":
                    org_map[rec_node["skos:prefLabel"]] = entity_id

            for rel_node in rel_list:
                rdf_frag += f';\n  {rel_node["pred"]} {rel_node["obj"]} '
                #print("related", rel_node)

            rdf_frag += f";\n  rdf:Type strw:{ent_type.capitalize()} "
            rdf_frag += "\n."

            #print(rdf_frag)
            rdf_list.append(rdf_frag)

    # construct the RDF graph
    for record_id, rec in data_records.items():
        #print(rec)
        rec_type: str = f'strw:{rec["RECORD_TYPE"].capitalize()}'
        name: str = ""
        employer: str = ""
        urls: typing.List[ str ] = []

        if rec_type == "strw:Organization":
            name = rec["NAMES"][0]["PRIMARY_NAME_ORG"]

            if "LINKS" in rec:
                for url_dict in rec["LINKS"]:
                    for url in url_dict.values():
                        urls.append(url)

            if "WEBSITE_ADDRESS" in rec:
                urls.append(rec["WEBSITE_ADDRESS"])

        else:
            if "NAME_FIRST" in rec:
                name = rec["NAME_FIRST"]

            if "NAME_MIDDLE" in rec:
                name += " " + rec["NAME_MIDDLE"]

            if "NAME_LAST" in rec:
                name += " " + rec["NAME_LAST"]

            if "SOURCE_LINKS" in rec:
                for url_dict in rec["SOURCE_LINKS"]:
                    for url in url_dict.values():
                        urls.append(url)

            if "EMPLOYER_NAME" in rec:
                org_name: str = rec["EMPLOYER_NAME"]

                if org_name in org_map:
                    employer = org_map[org_name]

        rdf_frag = f"{record_id} rdf:Type {rec_type} "
        rdf_frag += f";\n  skos:prefLabel \"{name}\"@\"{LANG_EN}\" "

        for url in urls:
            rdf_frag += f";\n  dct:identifier <{url}> "

        #print(rdf_frag, employer)
        rdf_list.append(rdf_frag)

        if len(employer) > 0:
            rdf_frag = f"{parent[record_id]} org:memberOf {employer} ."
            rdf_list.append(rdf_frag)

    print("\n".join(rdf_list))
