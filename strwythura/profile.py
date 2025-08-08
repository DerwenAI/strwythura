#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Profiling utilities.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import linecache
import tracemalloc
import typing

from pyinstrument import Profiler

KILO_B: float = 1024.0


class PerfProfiler:
    """
Profiling utilities for:

  * probablistic call trace
  * memory usage
    """

    def __init__ (
        self,
        ) -> None:
        """
Constructor.
        """
        self.profiler: Profiler = Profiler()


    def start (
        self,
        ) -> None:
        """
Start the profiling.
        """
        self.profiler.start()
        tracemalloc.start()


    def display_top (
        self,
        *,
        full: bool = False,
        key_type: str = "lineno",
        limit: int = 10,
        ) -> None:
        """
Display the top `limit` lines allocating the most memory,
ignoring `<frozen importlib._bootstrap>` and `<unknown>` files.
        """
        amount: tuple = tracemalloc.get_traced_memory()
        peak: float = round(amount[1] / KILO_B / KILO_B, 2)
        print(f"Peak memory usage: {peak} MB")

        if full:
            snapshot: tracemalloc.Snapshot = tracemalloc.take_snapshot().filter_traces((
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
            ))

            print(f"Top {limit} lines for memory usage:")
            top_stats: list = snapshot.statistics(key_type)

            for index, stat in enumerate(top_stats[:limit], 1):
                frame: tracemalloc.Frame = stat.traceback[0]

                print(
                    "#%s: %s:%s: %.1f KiB"  # pylint: disable=C0209
                    % (index, frame.filename, frame.lineno, stat.size / KILO_B)
                )

                line: typing.Optional[ str ] = linecache.getline(
                    frame.filename,
                    frame.lineno,
                )

                if line is not None:
                    print(f"    {line.strip()}")

            other: typing.Optional[ list ] = top_stats[limit:]

            if other is not None:
                size: float = sum(stat.size for stat in other)
                print(f"{len(other)} other usage: {round(size / KILO_B, 1)} KiB")

            total: float = sum(stat.size for stat in top_stats)
            print(f"Total allocated size: {round(total / KILO_B, 1)} KiB")


    def report (
        self,
        ) -> None:
        """
Stop the call trace profiler from further sampling,
take a snapshot of the memory usage, then report
performance statistics.
        """
        # call trace
        self.profiler.stop()
        self.profiler.print()

        # memory usage
        self.display_top()
        tracemalloc.stop()
