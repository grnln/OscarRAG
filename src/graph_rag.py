"""Graph RAG.

The LLM writes SPARQL from the ontology, GraphDB runs it, and the LLM
formats the resulting rows into a natural-language answer.
"""
from __future__ import annotations

import contextvars
import os
import re
from functools import lru_cache

# Set by OllamaForSparql at the end of every SPARQL-generation call.
# wsgi.ask_graph reads this to include the query in the JSON response.
last_sparql: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "last_sparql", default=None
)

from dotenv import load_dotenv
from langchain_community.chains.graph_qa.ontotext_graphdb import OntotextGraphDBQAChain
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

# Small models like to wrap SPARQL in ```sparql ... ``` fences. This strips them.
_FENCE_RE = re.compile(r"^```(?:sparql)?\s*\n?|\n?```\s*$", re.MULTILINE)

# Few-shot prompt for SPARQL generation. Overrides LangChain's default template.
# Concrete examples teach small LLMs (like Qwen 7B) our query patterns better
# than any amount of rule text.
SPARQL_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"],
    template="""Write a SPARQL SELECT query for the RDF graph described by the schema below.

Ontology schema (Turtle):
```
{schema}
```

Rules:
- Use only classes and properties from the schema.
- Include PREFIX declarations at the top.
- Individuals in this dataset are NOT typed with rdf:type. Identify entities
  by the predicates they use (e.g. films by :directedBy, people by rdfs:label).
- Prefer direct IRIs (e.g. :BestPicture) over string label filters when possible.
- Return ONLY the SPARQL query — no markdown, no explanation.

Examples:

Question: Who won Best Picture?
SPARQL:
PREFIX : <http://oscars2026.org#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?title WHERE {{
  ?film :wonAward :BestPicture ;
        rdfs:label ?title .
}}

Question: Which films did Ryan Coogler direct?
SPARQL:
PREFIX : <http://oscars2026.org#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?title WHERE {{
  ?film :directedBy :RyanCoogler ;
        rdfs:label ?title .
}}

Question: How many nominations did Sinners get?
SPARQL:
PREFIX : <http://oscars2026.org#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?n WHERE {{
  ?film rdfs:label "Sinners" ;
        :totalNominations ?n .
}}

Question: Who was nominated for Best Actor?
SPARQL:
PREFIX : <http://oscars2026.org#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name WHERE {{
  ?nominee :nominatedFor :BestActor ;
           rdfs:label ?name .
}}

Question: Which films won more than 3 awards?
SPARQL:
PREFIX : <http://oscars2026.org#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?title ?wins WHERE {{
  ?film :totalWins ?wins ;
        rdfs:label ?title .
  FILTER(?wins > 3)
}}

Question: {prompt}
SPARQL:""",
)

# QA prompt for stage 3 — turns rdflib-formatted SPARQL rows into a clean answer.
# LangChain's default expects a "well-formatted context" but our rows arrive as
# `[(rdflib.term.Literal('X'),)]`. Small models choke on that unless we spell it out.
QA_PROMPT = PromptTemplate(
    input_variables=["context", "prompt"],
    template="""Generate a natural-language answer using ONLY the SPARQL query results below.

The results may look like Python rdflib objects (e.g. `[(rdflib.term.Literal('X'),)]`
or `[(rdflib.term.URIRef('http://…#Something'),)]`). Extract the string values
(the text inside `Literal(...)`, or the local name after the last `#` in a URI).

Rules:
- Base your answer strictly on the results. Do NOT use general knowledge.
- If the results are empty (`[]`), say you don't have that information.
- Answer in one clear, natural sentence.

Examples:

Results: [(rdflib.term.Literal('Ryan Coogler'),)]
Question: Who directed Sinners?
Answer: Ryan Coogler directed Sinners.

Results: [(rdflib.term.Literal('One Battle After Another'),)]
Question: Which film won Best Picture?
Answer: One Battle After Another won Best Picture.

Results: [(rdflib.term.URIRef('http://oscars2026.org#PaulThomasAnderson'),)]
Question: Who won Best Director?
Answer: Paul Thomas Anderson won Best Director.

Results: []
Question: Who won Best Editing?
Answer: I don't have that information.

Results:
{context}

Question: {prompt}
Answer:""",
)


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
            "3. Individuals are NOT typed with rdf:type. Identify entities by "
            "the predicates they use (e.g. films by :directedBy, people by rdfs:label). "
            "Do NOT write patterns like `?x a :Film` — they return zero rows.\n"
            "4. Prefer direct IRIs (e.g. :BestPicture) over string label filters.\n"
            "5. If the ontology doesn't cover the question, return a valid SPARQL "
            "query that yields no results — never invent predicates."
        ))
        result = super()._generate([system] + list(messages), stop, run_manager, **kwargs)
        for gen in result.generations:
            cleaned = clean_sparql(gen.message.content)
            gen.message = AIMessage(content=cleaned)
            if isinstance(gen, ChatGeneration):
                gen.text = cleaned
        # Expose the cleaned SPARQL so the API layer can return it in the response.
        if result.generations:
            last_sparql.set(result.generations[0].message.content)
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
        llm=llm,
        graph=graph,
        sparql_generation_prompt=SPARQL_PROMPT,
        qa_prompt=QA_PROMPT,
        allow_dangerous_requests=True,
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
