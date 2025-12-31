#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manage the natural language processing.

see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import typing
import warnings

from icecream import ic
import spacy

from .ctx import DomainContext
from .elem import Entity, EntityInstance, EntitySource, NounSpan


class Parser:
    """
Wrapper class for the `spaCy` NLP pipeline used to extract entities
and relations, based on `GLiNER`, `DSPy`, and textgraphs.

The default definitions are for American English usage. Subclass
to modify the definitions for other languages and dialects.
    """
    BASE_CONCEPT: str = "owl:Thing"

    STOP_WORDS: set[ str ] = set([
        "PRON.each",
        "PRON.he",
        "PRON.it",
        "PRON.she",
        "PRON.some",
        "PRON.someone",
        "PRON.that",
        "PRON.their",
        "PRON.they",
        "PRON.those",
        "PRON.we",
        "PRON.what",
        "PRON.which",
        "PRON.who",
        "PRON.you",
    ])

    POS_TRANSFORM: dict[ str, str ] = {
        "PROPN": "NOUN",
    }


    def __init__ (
        self,
        config: dict,
        ctx: DomainContext,
        ) -> None:
        """
Constructor.
        """
        self.config: dict = config
        self.ctx: DomainContext = ctx
        self.label_map: dict[ str, str ] = self.ctx.get_label_map()
        self.ner_pipe: spacy.Language = self.build_ner_pipe()


    def build_ner_pipe (
        self,
        *,
        use_gliner: bool = True, # False
        ) -> spacy.Language:
        """
Initialize the `spaCy` pipeline used for NER + RE, by loading models
for `spaCy`, `GLiNER`

  - `ner_labels`: semantics to apply for zero-shot NER

This assumes the `spaCy` model has been downloaded already.

Note: this may take several minutes when run the first time after
installing the repo.
        """
        ner_pipe: spacy.Language = spacy.load(self.config["nlp"]["spacy_model"])

        if use_gliner:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                ner_pipe.add_pipe(
                    "gliner_spacy",
                    config = {
                        "style": "ent",
                        "labels": list(self.label_map.keys()),
                        "gliner_model": self.config["nlp"]["gliner_model"],
                        "chunk_size": self.config["vect"]["chunk_size"],
                    },
                )

        return ner_pipe


    def normalize_pos (
        self,
        pos: str,
        ) -> str:
        """
Normalize the _part-of-speech_ for downstream usage.
        """
        if pos in self.POS_TRANSFORM:
            return self.POS_TRANSFORM[pos]

        return pos


    def tokenize_lemma (
        self,
        span: list,
        *,
        debug: bool = False,
        ) -> str:
        """
Construct a parsed, lemmatized key for the given noun phrase.
        """
        lemma_key: str = " ".join([
            f"{self.normalize_pos(tok.pos_)}.{tok.lemma_.strip().lower()}"
            for tok in span
        ])

        if debug:
            ic(lemma_key, span)

        return lemma_key


    @classmethod
    def within_loc (
        cls,
        loc0: tuple[ int, int ],
        loc1: tuple[ int, int ],
        ) -> bool:
        """
Test whether the `loc0` tuple is within the `loc1` tuple.
        """
        return loc0[0] >= loc1[0] and loc0[1] <= loc1[1]

    @classmethod
    def overlaps_loc (
        cls,
        loc0: tuple[ int, int ],
        loc1: tuple[ int, int ],
        ) -> bool:
        """
Test whether the `loc0` tuple overlaps the `loc1` tuple.
        """
        if loc0[0] == loc0[1] or loc1[0] == loc1[1]:
            return False

        head: bool = loc0[0] >= loc1[0] and loc0[0] <= loc1[1]
        tail: bool = loc0[1] >= loc1[0] and loc0[1] <= loc1[1]

        return head ^ tail


    def transform_sentence (
        self,
        sent: spacy.tokens.span.Span,
        *,
        debug: bool = False,
        ) -> typing.Iterator[ NounSpan ]:
        """
Transform a parsed sentence into a sequence of the following, in order
of priority:

  * NER spans
  * noun chunks
  * tokens
        """
        nck_list: list[ NounSpan ] = [
            NounSpan(
                loc = ( nck.start, nck.end - 1, ),
                text = nck.text,
                span = nck,  # type: ignore
                source = EntitySource.NC,
                iri = self.BASE_CONCEPT,
            )
            for nck in sent.noun_chunks
            if len(nck) > 1
        ]

        if debug:
            ic(nck_list)

        ner_list: list[ NounSpan ] = [
            NounSpan(
                loc = ( ner.start, ner.end - 1, ),
                text = ner.text,
                span = ner,  # type: ignore
                label = ner.label_,
                source = EntitySource.NER,
                iri = self.label_map[ner.label_],
            )
            for ner in sent.ents
        ]

        if debug:
            ic(ner_list)

        tok_list: list[ NounSpan ] = [
            NounSpan(
                loc = ( tok.i, tok.i, ),
                text = tok.text.strip(),
                span = [ tok ],
                label = self.normalize_pos(tok.pos_),
                iri = self.BASE_CONCEPT,
            )
            for tok in sent
        ]

        # consolidate the noun chunks with named entities,
        # producing a list of spans
        span_list: list[ NounSpan ] = []

        todo: tuple | None = None

        while len(nck_list) > 0 and len(ner_list) > 0:
            last_todo: tuple | None = todo
            todo = ( len(nck_list), len(ner_list), )

            if debug:
                ic(todo, last_todo)

            if todo == last_todo:
                print("NCK list", nck_list)
                print("NER list", ner_list)
                raise RuntimeError("infinite loop")

            if debug:
                print("todo", todo, sent)

            nck: NounSpan = nck_list[0]
            ner: NounSpan = ner_list[0]

            if debug:
                ic(nck.loc, ner.loc)
                ic(self.within_loc(nck.loc, ner.loc))
                ic(self.overlaps_loc(nck.loc, ner.loc))

            #if nck.loc[1] == ner.loc[0]:
            if self.overlaps_loc(nck.loc, ner.loc):
                # NCK overlaps an NER
                nck_list.pop(0)

                ner_list.pop(0)
                span_list.append(ner)

            elif self.within_loc(nck.loc, ner.loc):
                # NER subsumes NCK
                if debug:
                    print("NER subsumes NCK", ner)

                while len(nck_list) > 0 and ner.loc[0] <= nck_list[0].loc[0] and ner.loc[1] >= nck_list[0].loc[1]:
                    nck_list.pop(0)

                ner_list.pop(0)
                span_list.append(ner)

            elif ner.loc[1] < nck.loc[0]:
                # emit NER
                if debug:
                    print("emit NER", ner)

                ner_list.pop(0)
                span_list.append(ner)

            elif ner.loc[0] > nck.loc[1]:
                # emit unlabeled NC
                if debug:
                    print("emit NCK", nck)

                nck_list.pop(0)
                span_list.append(nck)

            elif self.within_loc(ner.loc, nck.loc):
                # NCK subsumes NER, it inherits the label, iri, etc.
                if debug:
                    print("NCK subsumes NER", nck)

                ner_list.pop(0)
                nck.source = ner.source

                if nck.label is None:
                    nck.label = ner.label

                if nck.iri is None:
                    nck.iri = ner.iri

        span_list.extend(nck_list)
        span_list.extend(ner_list)
        span_list.sort(key = lambda span: span.loc)

        # consolidate the span list with tokens, producing a full list of parsed items
        full_list: list[ NounSpan ] = []

        while len(span_list) > 0 and len(tok_list) > 0:
            span: NounSpan = span_list[0]
            tok: NounSpan = tok_list[0]

            if tok.loc[0] < span.loc[0]:
                # emit token
                tok_list.pop(0)
                full_list.append(tok)

            elif tok.loc[0] >= span.loc[0] and tok.loc[0] <= span.loc[1]:
                # span subsumes token
                tok_list.pop(0)

            elif tok.loc[0] > span.loc[1]:
                # emit span
                span_list.pop(0)
                full_list.append(span)

        full_list.extend(span_list)
        full_list.extend(tok_list)
        full_list.sort(key = lambda span: span.loc)

        yield from full_list


    def parse_para (
        self,
        chunk_id: int,
        chunk_text: str,
        *,
        debug: bool = False,
        ) -> int:
        """
Parse a text paragraph, then per sentence:

  * transform into a sequence of NER spans, noun chunks, or tokens
  * extract entities, with labels mapped to IRIs where possible
  * tokenize entities based on lemmatization
  * use `textgraph` to add to the `LexicalGraph` lexical graph

For the paragraph, load an entity sequence vector into `gensim.Word2Vec`
using `EntityStore`
        """
        doc: spacy.tokens.doc.Doc = self.ner_pipe(chunk_text)
        num_sent: int = 0
        ent_seq: list[ Entity ] = []

        # transform sentence as: NER spans, noun chunks, tokens
        for sent_id, sent in enumerate(doc.sents):
            num_sent = sent_id
            sent_text: str = str(sent).strip()

            if debug:
                ic(sent_id, sent_text)

            ent_inst: EntityInstance = EntityInstance(
                sent_id = sent_id,
                chunk_id = chunk_id,
            )

            if debug:
                ic(ent_inst)

            for item in self.transform_sentence(sent, debug = debug):
                if debug:
                    ic(ent_seq, item)

                if item.source == EntitySource.LEX and item.label != "NOUN":
                    continue

                # tokenize entities based on lemmatization
                lemma_key: str = self.tokenize_lemma(item.span)

                if lemma_key in self.STOP_WORDS:
                    continue

                ent: Entity = Entity(
                    span = item,
                    lemma_key = lemma_key,
                )

                found_ent: Entity = self.ctx.ent_store.encode_entity(  # type: ignore
                    ent,
                    create = True,
                )

                found_ent.inst.append(ent_inst)
                ## TODO: make sequence based on paragraph
                ent_seq.append(found_ent)

            if debug:
                ic(sent_id, sent, ent_seq)

            # add this sentence to the _textgraph_ in `LexicalGraph`
            self.ctx.lex.add_sent(ent_seq)

        # load an entity sequence vector into `gensim.Word2Vec` using `EntityStore`
        seq_vec: list[ int ] = [
            ent.uid
            for ent in ent_seq
            if ent.uid is not None
        ]

        self.ctx.ent_store.embed_sequence(seq_vec)

        if debug:
            print(seq_vec)

        return num_sent + 1
