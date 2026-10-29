# RAG Assistant for Tightening Systems

A fully local assistant for a tightening-systems (screwdriving) plant. It answers questions from two sources and picks the right one automatically:

- **Documentation (RAG):** manuals, safety procedures, incident reports, machine configurations and a Bosch Rexroth tightening-systems catalog (PDF with dense tables). PDFs are parsed with Docling, tables are kept as separate chunks, and retrieval is hybrid (dense + sparse) in Qdrant. Answers are generated only from the retrieved chunks, with sources shown.
- **Live machine data (tool-calling agent):** a LangGraph ReAct agent calls a REST API for machine status, torque readings and error counts.
- **Router:** an LLM classifier sends each question to one of the two paths.

Everything runs locally (Ollama, Qdrant in Docker, Streamlit). No API keys, no data leaves the machine.

> The machine data, machine configurations, incident reports and error codes (E-460, E-330 and others) are **simulated** for this demo. The catalog PDF is a real, publicly available document and is not included in this repository (see Setup).

## Architecture

```
                      +-------------------+
 user question -----> |  Router (LLM)     |
                      +---------+---------+
                                |
               +----------------+----------------+
               |                                 |
          DOCS route                        data route*
               |                                 |
   +-----------v------------+        +-----------v-----------+
   |  RAG pipeline          |        |  Data agent (ReAct)   |
   |  Docling parsing       |        |  tools -> REST API    |
   |  Qdrant hybrid search  |        |                       |
   |  grounded local LLM    |        |                       |
   +-----------+------------+        +-----------+-----------+
               |                                 |
        answer + sources                  answer from tool results
```

\*The route is called `SQL` in the code for historical reasons. It queries a REST API, not a SQL database.

## How the RAG part works

1. **Parsing:** `.txt` files are read as text. PDFs are converted to Markdown with Docling (layout-aware, tables come out as Markdown tables).
2. **Chunking:** prose is split by size (800 characters, 100 overlap for text files). In PDFs, every table becomes its own chunk and gets the nearest heading as a prefix, so exact values and component IDs stay intact.
3. **Embedding:** each chunk gets a dense vector (`nomic-embed-text`) and a sparse BM25 vector (FastEmbed). Dense search finds meaning, sparse search finds exact tokens such as `3W050` or order numbers.
4. **Retrieval:** Qdrant runs hybrid search and fuses both rankings (reciprocal rank fusion). The top 4 chunks are used.
5. **Answer:** a local LLM (`qwen2.5:7b`) answers only from those chunks and refuses when the answer is not in them.

## Tech stack

- Python, LangChain, LangGraph
- Ollama: `qwen2.5:7b` (answers, routing), `llama3.2` (data agent), `nomic-embed-text` (embeddings)
- Qdrant (Docker) with dense + sparse vectors
- Docling for PDF parsing, FastEmbed for BM25 sparse vectors
- Streamlit for the chat interface
- Mock REST API built with Python's `http.server`

## Project structure

```
.
├── docs/                      knowledge base
│   ├── maintenance_manual.txt
│   ├── safety_procedures.txt
│   ├── incident_reports.txt
│   ├── machine_configuration.txt
│   └── (rexroth_tightening_spindles.pdf, not in repo)
├── rag_demo.py                load, parse, chunk, embed, retrieve, answer
├── data_agent.py              tool-calling agent for machine data
├── mock_api.py                simulated machine API (port 8000)
├── hybrid_router.py           routes each question to RAG or the agent
├── app.py                     Streamlit chat UI
├── eval_rag.py                scored evaluation (answer accuracy + retrieval hit rate)
├── test_rag.py                retrieval and refusal checks
├── test_router.py             routing checks
└── requirements.txt
```

## Setup

1. **Models** (install [Ollama](https://ollama.com) first):

   ```bash
   ollama pull qwen2.5:7b
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **Python environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Qdrant** (Docker):

   ```bash
   docker run -d --name qdrant -p 6333:6333 \
     -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
   ```

4. **Catalog PDF (optional):** download the Bosch Rexroth tightening systems catalog and save it as `docs/rexroth_tightening_spindles.pdf`. Without it the project runs on the txt files only, and the catalog questions in `eval_rag.py` will fail.

## Usage

```bash
python rag_demo.py        # builds the index in Qdrant, type "quit" when ready
python mock_api.py        # separate terminal: simulated machine API
streamlit run app.py      # chat UI at http://localhost:8501
```

Example questions:

| Question                                          | Path                           |
| ------------------------------------------------- | ------------------------------ |
| What does error E-460 mean?                       | documents                      |
| What is the order number of the 3W050 angle head? | documents (catalog table)      |
| Which angle head is installed on MCH-01?          | documents                      |
| What is the status of MCH-03?                     | machine data                   |
| What is the capital of France?                    | refused (not in documentation) |

## Evaluation

```bash
python eval_rag.py        # needs Ollama, Qdrant and the PDF
python test_rag.py        # rebuilds the index, so close the app first
python test_router.py
```

`eval_rag.py` runs questions with known answers and reports two numbers separately: **answer accuracy** (is the expected value in the answer?) and **retrieval hit rate** (was the expected source retrieved?). Separating them shows whether a failure comes from retrieval or from generation. Last run: 8/10 answers, 9/10 sources.

## Known limitations

- **Number grids in tables:** two evaluation questions fail (minimum circle diameter by spindle count, working range of the feed output drive). I have not yet determined whether the cause is table parsing or the small local model.
- **Small local models:** `qwen2.5:7b` and `llama3.2` are limited by the available laptop hardware and are weaker at table reading and tool calling than larger models.
- **One route per question:** a question that needs both documents and live data (for example "Is MCH-01's last torque above the limit of its angle head?") is not handled.
- **Data agent can overreach:** it may answer questions the API has no data for, such as "most downtime last month".
- **Sources:** answers show file name and chunk number. Page numbers are not shown yet.
- **Small test set:** ten questions are enough for debugging, not for measuring quality at scale.

## Next steps

- Page numbers from Docling's output as source references
- Reranker after hybrid search
- Row-wise splitting of long tables with repeated headers
- Excel, requirements and STEP file ingestion
- Run both paths for mixed questions and combine the answers
- Larger test set with per-document-type scores and latency measurements
