#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit example showing DSPy use.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import logging
import pathlib

import dspy  # type: ignore
import streamlit as st

from strwythura import GraphRAG, Workflow, \
    STRW_LOGO


@st.cache_resource
def load_assets (
    config_path: pathlib.Path,
    domain_path: pathlib.Path,
    *,
    run_local: bool = True,
    use_opik: bool = False,
    ) -> GraphRAG:
    """
Instantiate and configure the workflow manager, load the domain info,
and instantiate a `GraphRAG` object.
    """
    work: Workflow = Workflow(config_path = config_path)
    work.load_assets()
    work.ctx.open_vector_tables()
    work.load_parser()

    domain: dict = json.load(
        domain_path.open("r", encoding = "utf-8")
    )

    graph_rag: GraphRAG = GraphRAG(  # pylint: disable=W0621
        work,
        domain["name"],
        run_local = run_local,
        use_opik = use_opik,
    )

    return graph_rag


@st.fragment
def run_er_rag (
    graph_rag: GraphRAG,  # pylint: disable=W0621
    *,
    debug: bool = False,
    ) -> None:
    """
Main UI task as a `Streamlit.fragment`
    """
    st.title("Strwythura")

    # initialize chat history
    # display messages from history on app rerun
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show the question/response pair
    if question := st.chat_input("Quoi?"):
        with st.chat_message("user"):
            st.markdown(question)

        # enchanced GraphRAG prioritizes and retrieves text chunks
        chunks: list[ str ] = graph_rag.run_errag(
            question,
            debug = debug,
        )

        # LLM summarizes the text chunks in response to the question
        response: dspy.primitives.prediction.Prediction = graph_rag.qa_signature(
            question,
            chunks,
        )

        with st.chat_message("assistant", avatar = STRW_LOGO):
            st.markdown(response.response)

        # show the chat history
        st.divider()

        for message in st.session_state.messages:
            avatar: str | None = None

            if message["role"] == "assistant":
                avatar = STRW_LOGO.as_posix()

            with st.chat_message(message["role"], avatar = avatar):
                st.markdown(message["content"])

        # add the question and response to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.response,
        })


if __name__ == "__main__":
    logger: logging.Logger = logging.getLogger(__name__)
    logging.basicConfig(level = logging.WARNING) # DEBUG
    logger.info("set up, run only once")

    # interaction
    graph_rag: GraphRAG = load_assets(
        pathlib.Path("config.toml"),
        pathlib.Path("domain.json"),
    )

    run_er_rag(graph_rag)
