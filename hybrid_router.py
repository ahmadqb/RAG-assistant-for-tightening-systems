"""
Hybrid router: sends each question to either the SQL agent (live data)
or the RAG system (documentation).
"""
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from data_agent import run_data_agent

from rag_demo import CHROMA_DIR, EMBEDDING_MODEL, LLM_MODEL, ask_question
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode


# --- LOAD THE EXISTING VECTORSTORE (no rebuild) ---
def load_vectorstore():
    return QdrantVectorStore.from_existing_collection(
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        url="http://localhost:6333",
        collection_name="manuals",
    )





def run_sql_agent(question: str) -> str:
    try:
        return run_data_agent(question)
    except Exception as e:
        return f"Data agent error (is the API running at API_URL?): {e}"


# --- ROUTER ---
ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """Classify the user's question into exactly one category.

SQL: asks about live machine data: current status, torque values, which machines exist,
     or error counts and summaries across machines.
DOCS: asks about manuals, error code meanings, repair procedures, safety rules,
      PPE, lockout/tagout, written incident reports, or product catalog data such as
      component specs, order numbers, torque ratings, dimensions and weights.

Reply with only one word: SQL or DOCS.

Question: {question}
Category:"""
)


def route_question(question: str) -> str:
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    result = (ROUTER_PROMPT | llm).invoke({"question": question}).content.strip().upper()
    return "SQL" if "SQL" in result else "DOCS"


def answer(vectorstore, question: str):
    route = route_question(question)
    if route == "SQL":
        return route, run_sql_agent(question), []
    text, docs = ask_question(vectorstore, question)
    return route, text, docs


# --- MAIN ---
if __name__ == "__main__":
    vectorstore = load_vectorstore()
    print("HYBRID SYSTEM READY (type 'quit' to exit)")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        route, text, docs = answer(vectorstore, question)
        print(f"\n[Routed to: {route}]")
        print(text)
        if docs:
            print("\nSources:")
            for d in docs:
                print(f"  - {d.metadata['source']} (chunk {d.metadata['chunk_id']})")