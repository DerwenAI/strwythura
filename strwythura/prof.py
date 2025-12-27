#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A wrapper for the `pyInstrument` statistical call stack profiler.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import os

from psutil._common import bytes2human
import psutil
import pyinstrument


class Profiler:  # pylint: disable=R0903
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

        proc: psutil.Process = psutil.Process(os.getpid())
        print(f"memory used: {bytes2human(proc.memory_info().rss)}")
