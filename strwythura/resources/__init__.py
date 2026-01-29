#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Package definitions for the resource files.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import pathlib

import jinja2


STRW_LOGO: pathlib.Path = pathlib.Path(__file__).resolve().parent / "logo.png"
SZ_LOGO: pathlib.Path = pathlib.Path(__file__).resolve().parent / "senzing.png"

_jinja2_env: jinja2.Environment = jinja2.Environment(
    loader = jinja2.FileSystemLoader(
        pathlib.Path(__file__).resolve().parent
    )
)

PYVIS_JINJA_TEMPLATE: jinja2.Template = _jinja2_env.get_template("pyvis.jinja2")
