#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A wrapper for the `pyInstrument` statistical call stack profiler.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import pyinstrument


class Profiler:
    """
Use statistical call stack sampling to augment LLM observability.
    """

    def __init__(
        self,
        ) -> None:
        """
Constructor.
        """
        self.profiler: pyinstrument.Profiler = pyinstrument.Profiler()
        self.profiler.start()


    def analyze (
        self,
        ) -> None:
        """
Analyze and report the performance measures.
        """
        self.profiler.stop()
        self.profiler.print()
