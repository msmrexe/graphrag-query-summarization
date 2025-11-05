# GraphRAG: Global Query-Focused Summarization
 
This project implements the GraphRAG pipeline as described in the paper `"From Local to Global: A GraphRAG Approach to Query-Focused Summarization"` (Edge et al., 2024). It is a system designed to answer "global" questions about an entire document corpus by building a knowledge graph, identifying key communities, and synthesizing information hierarchically.
 
Developed for the M.S. System 2 AI course, this repository provides a clean, modular, and script-based implementation for document indexing and query-focused summarization.
 
## Features
 
* **Automated Document Ingestion:** Downloads and processes PDF documents, splitting them into manageable text chunks.
* **LLM-based Knowledge Graph Construction:** Uses a Large Language Model (LLM) to extract entities and relationships from text, building a comprehensive `networkx` graph.
* **Leiden Community Detection:** Implements the high-performance Leiden algorithm (via `igraph`) to partition the graph into semantically dense communities or topics.
* **Hierarchical Summarization:** Generates summaries for each community *at indexing time*, creating a "summary of summaries" to overcome context window limitations.
* **Global Query Answering:** Synthesizes a comprehensive answer to a user's query by intelligently combining insights from relevant community summaries.
* **Clean, Re-runnable Pipeline:** Separates the heavy **indexing** process from the lightweight **querying** process using distinct, CLI-driven scripts.
 
## Core Concepts & Techniques
 
* **Retrieval-Augmented Generation (RAG):** The standard RAG paradigm ("Retrieve then Read") excels at answering *local* questions by finding specific text chunks. However, it fails at *global* questions (e.g., "What are the main themes?") because no single chunk contains the answer.
* **Query-Focused Summarization (QFS):** The task of generating a summary of a document (or corpus) that is specifically tailored to a user's query. Traditional QFS methods struggle to scale to the large corpora managed by modern RAG systems.
* **GraphRAG (This Project):** This approach combines the strengths of both. Instead of a simple vector index of chunks, it builds a *structured graph index* of entities and relationships. This graph represents the entire corpus's knowledge. By detecting communities, it finds the "topics" of the corpus. It can then answer global questions by summarizing these topics, providing a "local to global" summarization.
 
---
 
## How It Works
 
This project's architecture is split into two main pipelines: **Indexing** (heavy, pre-computation) and **Querying** (light, real-time).
 
### 1. The Indexing Pipeline (`scripts/run_indexing.py`)
 
This pipeline is run once to process and index a new document.
 
1.  **Ingest & Chunk:** The source PDF is loaded (`PyPDFLoader`) and split into overlapping text chunks (`RecursiveCharacterTextSplitter`).
2.  **Entity & Relationship Extraction:** Each text chunk is fed to an LLM (e.g., Qwen) with a specific prompt to extract all entities and relationships (e.g., `"GraphRAG" -> "solves" -> "QFS scaling problem"`).
3.  **Knowledge Graph Construction:** The text-based extractions are parsed (using RegEx) and used to build a single, unified `networkx` knowledge graph. Nodes are entities (e.g., "RAG") and edges are the relationships between them (e.g., "is distinct from").
4.  **Community Detection (Leiden Algorithm):** This is the core of the "global" approach. A graph of thousands of nodes is too large to summarize. We use the **Leiden algorithm** to partition the graph into "communities"—clusters of nodes that are much more densely connected to each other than to the rest of the graph. Each community represents a distinct, coherent topic (e.g., one community for "RAG/QFS Methods," another for "LLM Authors").
    * **Implementation:** My implementation converts the `networkx` graph to an `igraph` object, creates a mapping from string node names to integer IDs, runs `leidenalg`, and then maps the integer-based communities back to their original string names.
5.  **Community Summarization:** The LLM is used *again* to generate a concise summary for each detected community.
6.  **Save Artifacts:** The final graph (`knowledge_graph.gml`) and list of community summaries (`community_summaries.json`) are saved to the `output/` directory.
 
### 2. The Query Pipeline (`scripts/run_query.py`)
 
This lightweight script is run every time a user asks a question.
 
1.  **Load Summaries:** The script loads the pre-computed `community_summaries.json` file.
2.  **Generate Intermediate Answers:** The user's query is passed to the LLM *in parallel* against **each** community summary. This generates dozens of "intermediate answers" (e.g., *Community 1's answer: "This community discusses RAG..."*, *Community 2's answer: "This community is not relevant..."*).
3.  **Synthesize Global Answer:** All intermediate answers are collected and fed to the LLM *one last time* with a synthesis prompt (e.g., "Combine these answers into one cohesive response"). This final step assembles the "local" pieces of information into the final "global" answer.
 
### 3. Analysis of Results
 
The key problem this system solves is the **failure of traditional RAG on global-scale questions**.
 
* **Traditional RAG Failures:** If you ask a normal RAG system "What problem does Graph RAG solve?", it might find the *introduction* chunk and the *conclusion* chunk. But it would fail to synthesize the nuanced relationships between RAG, QFS, and context windows discussed in the *middle* of the paper. It cannot see the "whole picture."
 
* **GraphRAG's Solution:** Our pipeline (simulating the one in the notebook) produced a final answer: "Graph RAG aims to solve the problem of limited context understanding and information retrieval in traditional RAG and QFS methods by leveraging graph structures to better capture and utilize relationships between entities."
 
This answer is comprehensive *because* it's not from one chunk. It's a synthesis:
1.  One community (e.g., "RAG vs. QFS") provided the "limited context understanding" part.
2.  Another community (e.g., "Graph Theory") provided the "leverage graph structures" part.
3.  A third (e.g., "Implementation Details") provided the "capture relationships" part.
 
By summarizing communities *at indexing time*, GraphRAG effectively compresses the entire corpus into a set of topical summaries. This overcomes the LLM's context window limit, allowing it to "reason" over the entire document by reading the pre-computed summaries instead of all the raw text.
 
---
 
## Project Structure
 
```
graphrag-query-summarization/
├── .gitignore               # Standard Python gitignore
├── LICENSE                  # MIT License
├── README.md                # This file
├── guided_run.ipynb         # Jupyter notebook for a guided demo
├── requirements.txt         # All Python dependencies
├── data/
│   └── .gitkeep             # Placeholder for input PDFs
├── logs/
│   └── .gitkeep             # Placeholder for .log files
├── output/
│   └── .gitkeep             # Placeholder for saved graphs & summaries
├── scripts/
│   ├── run_indexing.py      # Main script for the indexing pipeline (CLI)
│   └── run_query.py         # Main script for the query pipeline (CLI)
└── src/
    ├── __init__.py
    ├── data_processing.py   # Handles PDF download, loading, and splitting
    ├── graph_pipeline.py    # Core logic for graph build & community detection
    ├── llm_handler.py       # Manages loading and interacting with the LLM
    ├── query_handler.py     # Handles generation of final global answer
    └── utils.py             # Logging configuration
````
 
## How to Use
 
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/graphrag-query-summarization.git
    cd graphrag-query-summarization
    ````

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Indexing Pipeline (This will take time):**
    This script downloads the PDF, processes it, and saves the artifacts in the `output/` directory.
    ```bash
    python scripts/run_indexing.py
    ```

    *Note: You can specify a different Hugging Face model (e.g., a larger Qwen model) with the `--model_name` argument, but ensure you have the VRAM for it.*

4.  **Run the Query Pipeline:**
    Now you can ask global questions. The script will load the artifacts from `output/` to generate an answer.
    ```bash
    python scripts/run_query.py --query "What problem does Graph RAG aim to solve that traditional RAG cannot?"
    ```

5.  **Example Usage / Test the System:**
    ```bash
    # Example 1
    python scripts/run_query.py --query "How does GraphRAG use community detection in its pipeline?"
    
    # Example 2
    python scripts/run_query.py --query "Summarize the paper's main contributions."
    ```

---

## Author

Feel free to connect or reach out if you have any questions!

  * **Maryam Rezaee**
  * **GitHub:** [@msmrexe](https://github.com/msmrexe)
  * **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

-----

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for full details.
