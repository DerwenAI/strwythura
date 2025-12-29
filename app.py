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

from icecream import ic
import dspy  # type: ignore
import matplotlib.pyplot as plt
import polars as pl
import streamlit as st

from strwythura import GraphRAG, Workflow, \
    STRW_LOGO, SZ_LOGO


@st.cache_resource
def load_assets (
    config_path: pathlib.Path,
    domain_path: pathlib.Path,
    *,
    run_local: bool = True,
    use_opik: bool = True,
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


@st.fragment
def eval_buttons (
    ) -> None:
    """
Show like/nope evaluation buttons.
    """
    col1_, col2_, col3_ = st.columns([1 , 1, 12])

    with col1_:
        if did_like := st.button("", icon = ":material/thumb_up:"):
            eval_like()

    with col2_:
        if did_nope := st.button("", icon = ":material/thumb_down:"):
            eval_nope()

    with col3_:
        pass


def eval_like (
    ) -> None:
    """
User clicks a "thumb_up" button.
    """
    ic("like")


def eval_nope (
    ) -> None:
    """
User clicks a "thumb_down" button.
    """
    ic("nope")


def show_analytics (
    response: dspy.primitives.prediction.Prediction,
    df_perf: pl.DataFrame,  # pylint: disable=W0621
    ) -> None:
    """
Render analytics about the question/response sessions.
    """
    if len(df_perf) > 1:
        fig, ax1 = plt.subplots(1, 1)
        ax2 = ax1.twinx()

        ax1.plot(df_perf["tokens"], label = "tokens used", color = "green", linestyle = "dashed", marker = "o")
        ax1.set_ylabel("tokens used", color = "green")

        ax2.plot(df_perf["time"], label = "time (sec)", color = "blue", linestyle = "dotted", marker = "o")
        ax2.set_ylabel("processing time", color = "blue")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2)

        plt.xticks([])
        st.pyplot(fig)


    st.table(pl.DataFrame({
        "chunk_id": graph_rag.rag_chunks.keys(),  # pylint: disable=E0606
        "distance": graph_rag.rag_chunks.values(),
    }))

    st.table(graph_rag.anchor_nodes)
    st.write(response.get_lm_usage())


@st.fragment
def run_er_rag (
    graph_rag: GraphRAG,  # pylint: disable=W0621
    df_perf: pl.DataFrame,  # pylint: disable=W0621
    *,
    debug: bool = False,
    ) -> None:
    """
Main UI task as a `Streamlit.fragment`
    """
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
                df_perf.extend(pl.DataFrame({
                    "tokens": list(response.get_lm_usage().values())[0]["total_tokens"],
                    "time": time.time() - start_time,
                }))

            with st.chat_message("assistant", avatar = STRW_LOGO):
                st.markdown(response.response)
                eval_buttons()

            with st.expander("analytics"):
                show_analytics(
                    response,
                    df_perf,
                )

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
        initial_sidebar_state = st.session_state.get("sidebar_state", "expanded"),
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

    st.session_state.sidebar_state = "collapsed"

    # load assets
    graph_rag: GraphRAG = load_assets(
        pathlib.Path("config.toml"),
        pathlib.Path("domain.json"),
    )

    # render page
    st.header("Strwythura")
    st.text(f"Domain: {graph_rag.description}")
    st.divider()

    with st.sidebar:
        st.image(STRW_LOGO.as_posix())

        st.html("""
<strong>
<a
 href="https://pacoid.medium.com/strwythura-2a8007af3682?postPublishedType=repub"
 target="_blank"
 style="text-decoration: none;"
>tutorial</a>
<br/>
<a
 href="https://github.com/DerwenAI/strwythura/"
 target="_blank"
 style="text-decoration: none;"
>code repo</a>
<br/>
<a
 href="http://localhost:5173/"
 target="_blank"
 style="text-decoration: none;"
>Opik dashboard</a>
<br/>
<hr/>
<a
 href="https://senzing.com/gph-graph-rag-llm-knowledge-graphs/"
 target="_blank"
 style="text-decoration: none;"
>video</a>
<br/>
<a
 href="https://derwen.ai/s/2njz#1"
 target="_blank"
 style="text-decoration: none;"
>slides</a>
<br/>
<a
 href="https://derwen.ai/quiz/ai_def"
 target="_blank"
 style="text-decoration: none;"
>quiz</a>
<br/>
<a
 href="https://doi.org/10.5281/zenodo.17032671"
 target="_blank"
 style="text-decoration: none;"
>citation</a>
<br/>
<hr/>
<a
 href="https://senzing.com/graph-power-hour/"
 target="_blank"
 style="text-decoration: none;"
>Graph Power Hour!</a>
<br/>
<a
 href="https://senzing.com/"
 target="_blank"
 style="text-decoration: none;"
>Senzing home</a>
</strong>
        """)

        st.image(SZ_LOGO.as_posix())

    # interaction
    df_perf: pl.DataFrame = pl.DataFrame([
        pl.Series("tokens", [], dtype=pl.Int64),
        pl.Series("time", [], dtype=pl.Float64),
    ])

    run_er_rag(
        graph_rag,
        df_perf,
    )
