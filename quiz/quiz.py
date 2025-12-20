#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test rig"""

import hashlib
import json
import pathlib
import random

from icecream import ic
import jinja2


if __name__ == "__main__":
    quiz_path: pathlib.Path = pathlib.Path(__file__).parent.resolve() / "ai_def.json"

    # set up the Jinja2 environment and load the template
    env: jinja2.Environment = jinja2.Environment(loader = jinja2.FileSystemLoader("quiz"))
    template: jinja2.environment.Template = env.get_template("quiz.html")

    # render HTML using a JSON file
    with open(quiz_path, "r", encoding = "utf-8") as fp:
        dat: dict = json.load(fp)

    # shuffle the order of questions and choices
    for question in dat["questions"]:
        random.shuffle(question["choices"])

    random.shuffle(dat["questions"])

    # generate SHA-256 for the answer key
    dat["answer_key"] = []

    for question in dat["questions"]:
        for choice in question["choices"]:
            if choice["correct"]:
                message: str = question["name"] + choice["id"]
                hash_digest: str = hashlib.sha256(message.encode()).hexdigest()
                dat["answer_key"].append(hash_digest)

    # render the template
    html: str = template.render(
        quiz = dat,
    )

    print(html)
