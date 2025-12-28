#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit example showing DSPy use.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import json
import logging
import pathlib
import time

import dspy  # type: ignore
import polars as pl
import streamlit as st

from strwythura import GraphRAG, Workflow, \
    STRW_LOGO


DF_PERF: pl.DataFrame = pl.DataFrame([
    pl.Series("tokens", [], dtype=pl.Int64),
    pl.Series("time", [], dtype=pl.Float64),
])


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
load the assets, and instantiate a `GraphRAG` object.
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
        domain["description"],
        run_local = run_local,
        use_opik = use_opik,
    )

    return graph_rag


def show_analytics (
    response: dspy.primitives.prediction.Prediction,
    ) -> None:
    """
Render analytics about the question/response sessions.
    """
    global DF_PERF

    if len(DF_PERF) > 1:
        #st.pyplot(fig)
        st.table(DF_PERF)

    st.table(pl.DataFrame({
        "chunk_id": graph_rag.rag_chunks.keys(),
        "distance": graph_rag.rag_chunks.values(),
    }))

    st.table(graph_rag.anchor_nodes)
    st.write(response.get_lm_usage())


@st.fragment
def run_er_rag (
    graph_rag: GraphRAG,  # pylint: disable=W0621
    *,
    debug: bool = False,
    ) -> None:
    """
Main UI task as a `Streamlit.fragment`
    """
    global DF_PERF

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if question := st.chat_input("Quoi?"):
        col1, col2 = st.columns(2)

        # show the chat history
        with col2:
            for message in st.session_state.messages:
                avatar: str | None = None

                if message["role"] == "assistant":
                    avatar = STRW_LOGO.as_posix()

                with st.chat_message(message["role"], avatar = avatar):
                    st.markdown(message["content"])

        # show the question/response pair
        with col1:
            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner(text = "In progress...", show_time = True):
                start_time: float = time.time()

                # enchanced GraphRAG prioritizes and retrieves text chunks
                graph_rag.run_errag(
                    question,
                    debug = debug,
                )

                # LLM summarizes the text chunks in response to the question
                response: dspy.primitives.prediction.Prediction = graph_rag.qa_signature(
                    question,
                    graph_rag.get_chunks_text(),
                )

                # collect peformance statistics
                DF_PERF = pl.concat([
                    pl.DataFrame({
                        "tokens": list(response.get_lm_usage().values())[0]["total_tokens"],
                        "time": time.time() - start_time,
                    }),
                    DF_PERF,
                ])

            with st.chat_message("assistant", avatar = STRW_LOGO):
                st.markdown(response.response)

            with st.expander("analytics"):
                show_analytics(response)

        # add the question and response to chat history
        st.session_state.messages.insert(0, {
            "role": "assistant",
            "content": response.response,
        })

        st.session_state.messages.insert(0, {
            "role": "user",
            "content": question,
        })


if __name__ == "__main__":
    logger: logging.Logger = logging.getLogger(__name__)
    logging.basicConfig(level = logging.WARNING) # DEBUG
    logger.info("set up, run only once")

    # page set up
    st.set_page_config(
        page_title = "Strwythura",
        page_icon = STRW_LOGO.as_posix(),
        layout = "wide",
        initial_sidebar_state = "collapsed",
    )

    st.html("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&display=swap");

html, h1, h2, h3, h4, h5, p, span, cite, figcaption, button, input, select, textarea {
    font-family: "Atkinson Hyperlegible", sans-serif;
}

body {
    color: hsl(0, 0%, 40%);
}

p {
    font-weight: normal;
    font-size: 1em;
    line-height: 1.3em;
    margin: 1.2em 0 1.2em 0;
}
</style>
    """)

    # load assets
    graph_rag: GraphRAG = load_assets(
        pathlib.Path("config.toml"),
        pathlib.Path("domain.json"),
    )

    # render page
    st.header("Strwythura")
    st.text(graph_rag.description)
    st.divider()

    # interaction
    run_er_rag(graph_rag)
