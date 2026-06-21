# Client Success Stories Retrieval System

A production-grade, structure-aware, multimodal **Retrieval-Augmented Generation (RAG)** system exposed as a Model Context Protocol (MCP) server. 

Instead of searching standard text chunks, this system indexes documents by section hierarchies (proxy pointers) and uses multimodal synthesis (text + images/figures) to answer user queries with precise document citations and inline figure listings.

---

## How It Works

### 1. Ingestion & Indexing Pipeline
```
Document (PDF/DOCX/PPTX)
   │  Local Extraction (PyMuPDF / python-docx / python-pptx)
   ▼
Markdown + Image Figures (saved locally under data/)
   │
   ▼
Build Section Heading Tree (Section > Sub-section structure)
   │
   ▼
Generate Proxy Pointers (Breadcrumbs + Text body + Image anchors)
   │
   ▼
Database Ingestion (Embed breadcrumb+body -> PostgreSQL using pgvector)
```

1. **Extraction:** Documents synced from SharePoint are processed. Text is converted to markdown, and figures/charts are extracted and saved under `data/extracted_papers/<doc_id>/figures/`.
2. **Structure Tree:** The markdown document is parsed into a tree structure using heading levels (`#`, `##`, `###`). Each section node forms a **Proxy Pointer** mapping its structural path/breadcrumb (e.g. `Modernization > 3. Details > 3.1 Cost Savings`) to its content.
3. **Storage:** The section body and its breadcrumb path are embedded using Azure OpenAI Embeddings and stored in PostgreSQL in the `proxy_story_embeddings` table using `pgvector`.

---

### 2. Retrieval & Generation Pipeline
```
User Query ──► Embed ──► pgvector Cosine Search (Recall top 200 nodes)
                                   │
                                   ▼
                       Deduplicate Candidates (Top 50)
                                   │
                                   ▼
                       LLM Semantic Re-Rank (Top 5)
                                   │
                                   ▼
               Load Full Section Text + Related Figure Paths
                                   │
                                   ▼
                      Multimodal Vision Synthesis
               (Text Answer + Inline Citations + Figures)
```

1. **Broad Recall:** The user's query is embedded, and a cosine distance search (`<=>`) is run against PostgreSQL to pull the top `200` matching candidate sections.
2. **Deduplication:** Candidates are deduplicated by document and node IDs to narrow down to the top `50` unique sections.
3. **Semantic Re-ranking:** Candidate section breadcrumbs and snippets are sent to Azure OpenAI for re-ranking, picking the top `5` finalists (sections referencing requested figures or tables are prioritized).
4. **Multimodal Vision:** The full text of the finalist sections and their actual extracted figure images are passed to GPT-4o (Vision). The LLM synthesizes an answer that:
   - Ground claims using inline citations (e.g. `[1]`, `[2]`).
   - Cites figures inline using `[SHOW: filename | short caption]` format.
5. **Output Parsing:** The bot resolves the citations to their SharePoint download links and formats the answer for chatbot consumption.

---

## Directory Structure

* **[main.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/main.py):** Main Starlette HTTP entrypoint. Implements JWT authentication, SSE connection heartbeats, and message download endpoints.
* **[server.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/server.py):** Instantiates the FastMCP instance.
* **[tools/SSchatbot.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/tools/SSchatbot.py):** Registers MCP tools (e.g. `message_gpt`, session creation/renaming/soft-deletion) wrapping session histories and chatbot responses.
* **[utils/proxy_rag/](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/utils/proxy_rag):** Core modules of the Proxy Pointer RAG engine:
  * **[extract.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/utils/proxy_rag/extract.py):** Markdown converter and image parser.
  * **[tree_builder.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/utils/proxy_rag/tree_builder.py):** Construct section nodes and logs ledger files.
  * **[pg_vector_store.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/utils/proxy_rag/pg_vector_store.py):** Pool and connection setup to PostgreSQL with pgvector operations.
  * **[indexer.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/utils/proxy_rag/indexer.py):** Triggers extraction/embeddings and executes searches.
  * **[rag_bot.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/utils/proxy_rag/rag_bot.py):** MMRagBot recall and re-ranking routines.
* **[utils/rag_pipeline.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/utils/rag_pipeline.py):** Main interface adapter for `SSchatbot.py`.
* **[utils/rag_pipeline_index.py](file:///Users/shlok/Coding/success/success/src/success-stories-retrival-system/utils/rag_pipeline_index.py):** Indexing adapter to sync documents from SharePoint to the database.

---

## Setup & Steps to Run

### Step 1: Install Dependencies
Install all package requirements in your virtual environment:
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Create a `.env` file in `src/success-stories-retrival-system/`:
```env
# Database Settings
POSTGRES_DATABASE_CONNECTION_STRING=postgresql://<user>:<password>@<host>:5432/<database>

# Azure OpenAI LLM Settings
AZURE_OPENAI_LLM_MODEL_API_KEY=your_azure_llm_api_key
AZURE_OPENAI_LLM_MODEL_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_LLM_MODEL_LLM_MODEL=your_gpt_4o_deployment_name
AZURE_OPENAI_LLM_MODEL_API_VERSION=2024-12-01-preview

# Azure OpenAI Embedding Settings
AZURE_OPENAI_EMBEDDING_MODEL_API_KEY=your_azure_embeddings_api_key
AZURE_OPENAI_EMBEDDING_MODEL_API_BASE=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL=your_embeddings_deployment_name
AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION=2024-02-01

# SharePoint Integration Details
SHAREPOINT_CLIENT_ID=your_client_id
SHAREPOINT_CLIENT_SECRET=your_client_secret
SHAREPOINT_TENANT_ID=your_tenant_id
SHAREPOINT_SITE_HOSTNAME=your_site.sharepoint.com
SHAREPOINT_SITE_PATH=/sites/your-site
```

### Step 3: Run Document Ingestion (Index Stories)
To fetch your documents from SharePoint and parse them into the pgvector table, trigger the indexing pipeline. This can be run using a startup script or a Jupyter notebook cell:
```python
from utils.rag_pipeline_index import Index_Success_Stories
from utils.sharepoint_manager import SharePointManager

# Connect to SharePoint and run the indexer
sharepoint = SharePointManager(...)
indexer = Index_Success_Stories(sharepoint_manager=sharepoint)
indexer.index_all_stories(folder_path="General/Publish Ready Success Stories")
```
*Note: This automatically validates database connections and populates the `proxy_story_embeddings` table.*

### Step 4: Run the Server
Launch the HTTP server locally (which exposes the MCP endpoints on port 8000):
```bash
uvicorn main:http_app --reload --port 8000
```
The server will handle authorization, conversation history storage in MongoDB, and route user messages through the Proxy Pointer RAG engine.
