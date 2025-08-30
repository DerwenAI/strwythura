#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data mapping from CSV to JSONL for Senzing input

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import csv
import hashlib
import json
import pathlib
import sys
import typing

from icecream import ic


if __name__ == "__main__":
    dataset: typing.List[ dict ] = []
    csv_path: pathlib.Path = pathlib.Path(sys.argv[1])

    # load the CSV input data
    with open(csv_path.as_posix(), newline = "") as fp:
        headers: typing.List[ str ] = []
        reader: csv.reader = csv.reader(fp, delimiter = ",")

        for line_num, row in enumerate(reader):
            md: hashlib.shake_256 = hashlib.shake_256()

            for val in row:
                md.update(val.encode(encoding = "utf-8"))

            if line_num < 1:
                headers = row
            else:
                print(row)

                if csv_path.as_posix() in [ "acme_biz.csv", "corp_home.csv", ]:
                    rec_type: str = "ORGANIZATION"
                else:
                    rec_type = "PERSON"

                data: dict = {
                    "DATA_SOURCE": csv_path.stem.upper(),
                    "RECORD_TYPE": rec_type,
                    "RECORD_ID": md.hexdigest(20),
                }

                for col in headers:
                    val: str = row[headers.index(col)].strip()
                    ic(col, val)

                    if col == "name":
                        if rec_type == "ORGANIZATION":
                            data["NAMES"] = [{
                                "PRIMARY_NAME_ORG": val,
                            }]

                    elif col == "first name":
                        data["NAME_FIRST"] = val
                    elif col == "middle name":
                        if len(val) > 0:
                            data["NAME_MIDDLE"] = val
                    elif col == "last name":
                        data["NAME_LAST"] = val

                    elif col in [ "address", "location" ]:
                        data["ADDR_FULL"] = val

                    elif col == "affiliation":
                        data["EMPLOYER_NAME"] = val

                    elif col == "profile":
                        if rec_type == "ORGANIZATION":
                            label: str = "unknown"

                            if "opencorporates" in val:
                                label = "OpenCorporates"
                            elif "dnb.com" in val:
                                label = "DUNS"

                            data["LINKS"] = [{
                                label: val,
                            }]
                        else:
                            data["SOURCE_LINKS"] = [{
                                "SOURCE_URL": val
                            }]

                    elif col == "url":
                        data["WEBSITE_ADDRESS"] = val

                ic(data)
                dataset.append(data)

    # write the transformed data as a JSONL file
    with open(csv_path.with_suffix(".json"), "w") as fp:
        for data in dataset:
            line = json.dumps(data)
            fp.write(line)
            fp.write("\n")
