# Multimodal Debate RAG Engine

A full-stack, vision-enabled Retrieval-Augmented Generation (RAG) agent designed to ingest complex PDF documents, orchestrate automated debates, and render verifiable verdicts based on text and graphical evidence. 

Built as a final-year B.Tech project, this engine overcomes standard LLM context limitations by utilizing modality-isolated vector retrieval and autonomous agentic workflows.

## 🚀 Features

*   **Multimodal PDF Ingestion:** Parses dense academic or corporate PDFs, extracting both textual chunks and structural diagrams/charts (filtering out graphical noise).
*   **Page-Linked Modality Retrieval:** Circumvents CLIP's limitations with complex charts by dynamically linking extracted images to high-scoring text chunks based on document page proximity.
*   **Agentic Orchestration (LangGraph):** Uses a graph-based workflow to retrieve evidence, self-grade the context, fall back to web search if necessary, and finally synthesize a verdict.
*   **Automated Model Cascading:** Implements a robust API failover system that seamlessly switches between Gemini models (e.g., 3.8-flash, 3.5-flash-lite) to handle rate limits and quota exhaustion automatically.
*   **Dual-Pane UI:** A modern, Node-free FastAPI HTML template featuring real-time drag-and-drop document ingestion, side-by-side debate contrast parsing, and a fullscreen image lightbox.

## 🛠️ Tech Stack

*   **Backend:** FastAPI, Python
*   **Orchestration:** LangGraph
*   **Embeddings:** CLIP (OpenAI / HuggingFace)
*   **Vector Database:** Qdrant (Local instance)
*   **LLM Inference:** Google Gemini API (via REST)
*   **Document Processing:** PyMuPDF, Langchain Text Splitters
*   **Frontend:** HTML5, Tailwind CSS (via CDN)

## 🏗️ System Architecture

1.  **Ingestion (`ingest.py`):** Accepts a binary PDF stream, tears down the existing Qdrant collection, chunks text (800 char / 100 overlap), extracts valid images (>250px), embeds them via CLIP, and stores them in Qdrant.
2.  **Retrieval (`retrieve.py`):** Queries Qdrant using separate text and image modality filters to guarantee the presence of visual evidence without bloating the search limit.
3.  **Agent Workflow (`agent.py`):** A StateGraph that evaluates claims, checks context sufficiency, and formats the output into strict *Supporting*, *Refuting*, and *Final Verdict* components.
4.  **API & UI (`main.py` & `index.html`):** FastAPI serves the inference endpoints and mounts the Tailwind-powered aesthetic interface.

## 🚦 Local Setup

1. Clone the repository and navigate into the project directory.
2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Google Gemini API key:
   ```text
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Start the application:
   ```bash
   uvicorn main:app --reload --port 8001
   ```
5. Open your browser and navigate to `http://127.0.0.1:8001/` to access the UI.