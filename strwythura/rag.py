#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper for using DSPy to perform a RAG signature.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing

import dspy  # type: ignore


class DSPy_RAG (dspy.Module):  # pylint: disable=C0103
    """
DSPy implementation of a RAG signature.
    """

    def __init__(  # pylint: disable=W0231
        self,
        config: dict,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config

        # load the LLM
        self.lm: dspy.LM = dspy.LM(
            self.config["rag"]["lm_name"],
            api_base = self.config["rag"]["api_base"],
            api_key = "",
            temperature = self.config["rag"]["temperature"],
            max_tokens = self.config["rag"]["max_tokens"],
            stop = None,
            cache = False,
        )

        dspy.configure(
            lm = self.lm
        )

        self.respond: dspy.Predict = dspy.Predict(
            "context, question -> response"
        )

        self.context: typing.List[ str ] = []


    def forward (
        self,
        question: str,
        ) -> dspy.primitives.prediction.Prediction:
        """
Invoke the RAG signature.
        """
        reply: dspy.primitives.prediction.Prediction = self.respond(
            context = self.context,
            question = question,
        )

        return reply
