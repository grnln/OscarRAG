"""Graph RAG.

The LLM writes SPARQL from the ontology, GraphDB runs it, and the LLM
formats the resulting rows into a natural-language answer.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration
from langchain_ollama import ChatOllama

load_dotenv()

# Small models like to wrap SPARQL in ```sparql ... ``` fences. This strips them.
_FENCE_RE = re.compile(r"^```(?:sparql)?\s*\n?|\n?```\s*$", re.MULTILINE)


def clean_sparql(text: str) -> str:
    """Strip markdown code fences from a SPARQL string."""
    return _FENCE_RE.sub("", text.strip()).strip()


class OllamaForSparql(ChatOllama):
    """Ollama wrapped to pin the namespace prefix and strip fences — SPARQL requests only.

    LangChain uses the same LLM for two stages (SPARQL generation + answer
    formatting). We only want to intervene on the SPARQL stage; the answer
    stage should return plain prose. We detect a SPARQL request by looking
    for LangChain's SPARQL-generation prompt marker in the messages.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Prepend a SPARQL-writer system prompt + strip fences — only for SPARQL calls."""
        is_sparql = any(
            "Write a SPARQL" in getattr(m, "content", "") for m in messages
        )
        if not is_sparql:
            return super()._generate(messages, stop, run_manager, **kwargs)

        namespace = os.environ.get("GRAPHDB_NAMESPACE", "")
        system = SystemMessage(content=(
            "You are a SPARQL expert answering questions over an RDF graph.\n\n"
            "Rules:\n"
            f"1. Always start with: PREFIX : <{namespace}>\n"
            "2. Output ONLY the SPARQL query — no markdown, no explanation.\n"
            "3. Prefer direct IRIs (e.g. :BestPicture) over string label filters.\n"
            "4. If the ontology doesn't cover the question, return a valid SPARQL "
            "query that yields no results — never invent predicates."
        ))
        result = super()._generate([system] + list(messages), stop, run_manager, **kwargs)
        for gen in result.generations:
            cleaned = clean_sparql(gen.message.content)
            gen.message = AIMessage(content=cleaned)
            if isinstance(gen, ChatGeneration):
                gen.text = cleaned
        return result


def plain_llm() -> ChatOllama:
    """Plain chat LLM — used by Text RAG for free-form answers."""
    return ChatOllama(model=os.environ["OLLAMA_MODEL"], temperature=0)


@lru_cache(maxsize=1)
def build_chain() -> OntotextGraphDBQAChain:
    """Build (and cache) the Graph RAG chain: schema → SPARQL → GraphDB → answer."""
    graph = OntotextGraphDBGraph(
        query_endpoint=f"{os.environ['GRAPHDB_URL'].rstrip('/')}"
                       f"/repositories/{os.environ['GRAPHDB_REPOSITORY']}",
        local_file=os.environ["GRAPHDB_ONTOLOGY_FILE"],
    )
    llm = OllamaForSparql(model=os.environ["OLLAMA_MODEL"], temperature=0)
    return OntotextGraphDBQAChain.from_llm(
        llm=llm, graph=graph, allow_dangerous_requests=True,
    )


def ask(question: str) -> str:
    """Answer a natural-language question using the knowledge graph."""
    chain = build_chain()
    return chain.invoke({chain.input_key: question})[chain.output_key]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.graph_rag <question>")
        sys.exit(2)
    print(ask(" ".join(sys.argv[1:])))
