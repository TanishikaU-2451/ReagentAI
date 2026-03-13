# ReagentAI

**An autonomous AI engineering system that reads scientific research papers and generates executable machine learning implementations.**

ReagentAI ingests a PDF research paper, extracts its core ideas through a multi-agent LLM pipeline, retrieves reference code from GitHub and pre-trained models from HuggingFace, generates a complete ML project, executes it inside a Docker sandbox, iteratively debugs any failures, validates the output, scores its reproducibility, and packages everything for download -- all without human intervention.

---

## Key Features

- **PDF Paper Parsing** -- Extracts full text, metadata, section structure, and LaTeX equations from uploaded research papers using PyMuPDF and pdfplumber.
- **RAG Knowledge Engine** -- Chunks and embeds paper content into a FAISS vector store for semantic retrieval, powering context-aware agent prompts and paper chat.
- **Multi-Agent System (9 Agents)** -- Specialised LLM agents (Research, Planning, Architecture, Coding, Testing, Debug, Validation, Diagram, Chat) collaborate through an orchestrated pipeline, each using a model suited to its role.
- **GitHub Repository Retrieval** -- Extracts keywords from the paper and searches GitHub for reference implementations, ranks repositories by relevance, and extracts reusable code patterns.
- **HuggingFace Model Retrieval** -- Identifies model families mentioned in the paper and searches the HuggingFace Hub for relevant pre-trained checkpoints and libraries.
- **Code Generation** -- The CodingAgent produces a complete, runnable ML project (source files, configuration, requirements, README) based on the plan and architecture.
- **Docker Sandbox Execution** -- Generated projects are executed inside an isolated Docker container with enforced resource limits (memory, CPU, timeout) and network restrictions.
- **Self-Debug Loop** -- An iterative generate-execute-capture-fix cycle (up to 5 attempts) where the DebugAgent diagnoses runtime errors and patches the code automatically.
- **Validation** -- Static and runtime checks verify the generated project's correctness, structure, and adherence to the paper's specification.
- **Architecture Diagrams** -- Mermaid.js diagrams are generated to visualise the implemented system's architecture, data flow, and class relationships.
- **Reproducibility Scoring** -- A multi-category assessment scores how faithfully the generated code reproduces the paper's methodology, with actionable feedback.
- **Paper Chat** -- A RAG-augmented conversational interface lets users ask natural-language questions about the uploaded paper and receive cited answers.
- **Project Packaging** -- The final project (source code, tests, configs, diagrams, reports) is packaged into a downloadable `.zip` archive.

---

## Tech Stack

| Layer         | Technology                                                                                          |
|---------------|-----------------------------------------------------------------------------------------------------|
| Backend       | Python 3.11, FastAPI, Uvicorn, Pydantic                                                            |
| Frontend      | Next.js 14 (App Router), React, TypeScript, Tailwind CSS                                           |
| LLM Inference | HuggingFace Inference API -- Llama-3 8B, Mixtral 8x7B, CodeLlama 34B, StarCoder2                  |
| Embeddings    | `sentence-transformers/all-mpnet-base-v2` via SentenceTransformers                                 |
| Vector DB     | FAISS (`IndexFlatIP`, cosine similarity via L2-normalised inner product)                           |
| Sandbox       | Docker (Python 3.11-slim containers, resource-limited, network-isolated)                           |
| PDF Parsing   | PyMuPDF, pdfplumber, SymPy (LaTeX equation extraction)                                             |
| Logging       | Loguru (structured, per-agent log files)                                                           |
| HTTP          | httpx, aiohttp (async GitHub/HuggingFace API calls)                                               |
| Containerisation | Docker, Docker Compose                                                                          |

---

## Repository Structure

```
ReagentAI/
├── .env.example                          # Environment variable template
├── Dockerfile                            # Backend Docker image (multi-stage)
├── docker-compose.yml                    # Full-stack orchestration (backend + frontend + sandbox)
├── requirements.txt                      # Python dependencies
│
├── backend/
│   ├── main.py                           # FastAPI app entry point, REST + WebSocket endpoints
│   ├── config/
│   │   └── settings.py                   # Pydantic settings (env vars, model IDs, paths)
│   ├── agents/
│   │   ├── base_agent.py                 # Abstract base class, HuggingFace API call_model()
│   │   ├── research_agent.py             # Extracts structured summary from paper text
│   │   ├── planning_agent.py             # Converts research summary into implementation plan
│   │   ├── architecture_agent.py         # Designs project directory layout and modules
│   │   ├── coding_agent.py               # Generates source code files
│   │   ├── testing_agent.py              # Generates test cases
│   │   ├── debug_agent.py                # Diagnoses runtime errors and produces fixes
│   │   ├── validation_agent.py           # Runs validation checks on generated code
│   │   ├── diagram_agent.py              # Generates Mermaid.js architecture diagrams
│   │   └── chat_agent.py                 # RAG-augmented paper Q&A
│   ├── parser/
│   │   ├── pdf_parser.py                 # PDF text and metadata extraction
│   │   ├── section_extractor.py          # Section boundary detection
│   │   └── equation_extractor.py         # LaTeX equation extraction
│   ├── retrieval/
│   │   ├── rag_pipeline.py               # End-to-end chunking, embedding, indexing, retrieval
│   │   ├── vector_store.py               # FAISS IndexFlatIP wrapper with metadata sidecar
│   │   ├── embedding_model.py            # SentenceTransformer wrapper (lazy-loaded, batched)
│   │   ├── github_search.py              # GitHub API keyword search and repo retrieval
│   │   ├── huggingface_retrieval.py      # HuggingFace Hub model search
│   │   ├── repo_ranker.py                # Repository relevance ranking
│   │   └── code_extractor.py             # Reference code extraction from repositories
│   ├── execution/
│   │   ├── sandbox.py                    # Docker container lifecycle and sandboxed execution
│   │   └── code_runner.py                # Project execution orchestration
│   ├── orchestration/
│   │   ├── pipeline.py                   # Master 16-stage pipeline orchestrator
│   │   ├── code_generator.py             # File generation and disk-write pipeline
│   │   ├── debug_loop.py                 # Iterative generate-execute-debug cycle
│   │   ├── validator.py                  # Static + runtime validation pipeline
│   │   ├── diagram_generator.py          # Mermaid diagram generation orchestrator
│   │   ├── reproducibility.py            # Reproducibility scoring engine
│   │   ├── chat_handler.py               # Paper chat session manager
│   │   └── project_packager.py           # Zip archive packaging
│   ├── memory/
│   │   └── agent_memory.py               # FAISS-backed per-agent long-term memory store
│   └── utils/
│       └── logging.py                    # Loguru configuration and logger factory
│
├── frontend/
│   └── nextjs_app/
│       ├── Dockerfile                    # Frontend Docker image
│       ├── package.json                  # Node.js dependencies
│       ├── next.config.js                # Next.js configuration
│       ├── tailwind.config.js            # Tailwind CSS configuration
│       ├── tsconfig.json                 # TypeScript configuration
│       ├── app/
│       │   ├── layout.tsx                # Root layout
│       │   ├── page.tsx                  # Home page (upload)
│       │   └── dashboard/[id]/page.tsx   # Project dashboard (progress, code, diagrams, chat)
│       ├── components/
│       │   ├── UploadPaper.tsx           # PDF upload form with drag-and-drop
│       │   ├── AgentProgressViewer.tsx   # Real-time pipeline stage tracker
│       │   ├── CodeViewer.tsx            # Syntax-highlighted file explorer
│       │   ├── LogsPanel.tsx             # Live log stream panel
│       │   ├── ArchitectureDiagramViewer.tsx  # Mermaid diagram renderer
│       │   ├── ChatInterface.tsx         # Paper Q&A chat interface
│       │   └── DownloadProject.tsx       # Project zip download button
│       └── lib/
│           ├── api.ts                    # Typed API client for all backend endpoints
│           └── websocket.ts              # WebSocket hook with auto-reconnect
│
├── generated_projects/                   # Output directory for generated ML projects
├── vector_store/                         # Persisted FAISS indexes and metadata
├── logs/                                 # Application log files
└── docs/                                 # Additional documentation
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for sandbox execution)
- A [HuggingFace](https://huggingface.co/settings/tokens) API token
- A [GitHub](https://github.com/settings/tokens) personal access token (optional, increases rate limits)

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ReagentAI.git
cd ReagentAI
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your tokens:

```dotenv
HUGGINGFACE_API_TOKEN=hf_your_actual_token
GITHUB_TOKEN=ghp_your_actual_token
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend/nextjs_app
npm install
cd ../..
```

### 5. Start the Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

### 6. Start the Frontend

```bash
cd frontend/nextjs_app
npm run dev
```

The UI will be available at `http://localhost:3000`.

### 7. Or Use Docker Compose (Recommended)

```bash
docker-compose up --build
```

This starts the backend (port 8000), frontend (port 3000), and sandbox container automatically.

---

## API Endpoints

| Method | Endpoint                  | Description                                  |
|--------|---------------------------|----------------------------------------------|
| POST   | `/api/upload`             | Upload a PDF research paper                  |
| GET    | `/api/projects/{id}`      | Get project info and pipeline status         |
| GET    | `/api/code/{id}`          | Get the generated project file tree          |
| GET    | `/api/code/{id}/file`     | Get content of a specific generated file     |
| GET    | `/api/diagrams/{id}`      | Get Mermaid.js architecture diagrams         |
| GET    | `/api/validation/{id}`    | Get validation results and check details     |
| GET    | `/api/score/{id}`         | Get reproducibility score and feedback       |
| POST   | `/api/chat/{id}`          | Send a chat message about the paper          |
| GET    | `/api/download/{id}`      | Download the generated project as a zip      |
| WS     | `/ws/pipeline/{id}`       | Real-time pipeline progress stream           |
| GET    | `/health`                 | Backend health check                         |

---

## Environment Variables

| Variable                   | Default                                        | Description                                 |
|----------------------------|------------------------------------------------|---------------------------------------------|
| `HUGGINGFACE_API_TOKEN`    | _(required)_                                   | HuggingFace Inference API token             |
| `GITHUB_TOKEN`             | _(optional)_                                   | GitHub personal access token                |
| `BACKEND_HOST`             | `0.0.0.0`                                      | Backend listen address                      |
| `BACKEND_PORT`             | `8000`                                         | Backend listen port                         |
| `NEXT_PUBLIC_API_URL`      | `http://localhost:8000`                        | Frontend API base URL                       |
| `SANDBOX_IMAGE`            | `python:3.11-slim`                             | Docker image for sandbox execution          |
| `SANDBOX_TIMEOUT`          | `300`                                          | Max execution time in seconds               |
| `SANDBOX_MEMORY_LIMIT`     | `2g`                                           | Docker memory limit                         |
| `VECTOR_STORE_PATH`        | `./vector_store`                               | FAISS index persistence directory           |
| `EMBEDDING_MODEL`          | `sentence-transformers/all-mpnet-base-v2`      | Embedding model for RAG                     |
| `RESEARCH_AGENT_MODEL`     | `meta-llama/Meta-Llama-3-8B-Instruct`          | Research agent LLM                          |
| `PLANNING_AGENT_MODEL`     | `mistralai/Mixtral-8x7B-Instruct-v0.1`        | Planning agent LLM                          |
| `ARCHITECTURE_AGENT_MODEL` | `mistralai/Mixtral-8x7B-Instruct-v0.1`        | Architecture agent LLM                      |
| `CODING_AGENT_MODEL`       | `meta-llama/CodeLlama-34b-Instruct-hf`        | Coding agent LLM                            |
| `DEBUG_AGENT_MODEL`        | `meta-llama/CodeLlama-34b-Instruct-hf`        | Debug agent LLM                             |
| `TESTING_AGENT_MODEL`      | `meta-llama/CodeLlama-34b-Instruct-hf`        | Testing agent LLM                           |
| `VALIDATION_AGENT_MODEL`   | `meta-llama/CodeLlama-34b-Instruct-hf`        | Validation agent LLM                        |
| `DIAGRAM_AGENT_MODEL`      | `mistralai/Mixtral-8x7B-Instruct-v0.1`        | Diagram agent LLM                           |
| `CHAT_AGENT_MODEL`         | `meta-llama/Meta-Llama-3-8B-Instruct`          | Chat agent LLM                              |
| `LOG_LEVEL`                | `DEBUG`                                        | Logging verbosity                           |
| `LOG_PATH`                 | `./logs`                                       | Log file directory                          |
| `MAX_DEBUG_ATTEMPTS`       | `5`                                            | Max self-debug loop iterations              |
| `GENERATED_PROJECTS_PATH`  | `./generated_projects`                         | Output directory for generated projects     |

---

## Agent Model Roles

| Agent          | Role                                                  | Default Model                                 |
|----------------|-------------------------------------------------------|-----------------------------------------------|
| Research       | Extract structured summary from paper (algorithm, architecture, datasets, hyperparameters, equations) | `meta-llama/Meta-Llama-3-8B-Instruct` (Llama-3) |
| Planning       | Convert research summary into a step-by-step implementation plan | `mistralai/Mixtral-8x7B-Instruct-v0.1` (Mixtral) |
| Architecture   | Design the project directory layout, module structure, and class hierarchy | `mistralai/Mixtral-8x7B-Instruct-v0.1` (Mixtral) |
| Coding         | Generate complete source code files from the plan and architecture | `meta-llama/CodeLlama-34b-Instruct-hf` (CodeLlama) |
| Debug          | Diagnose runtime errors and produce corrected code patches | `meta-llama/CodeLlama-34b-Instruct-hf` (CodeLlama) |
| Testing        | Generate unit tests and integration test cases         | `meta-llama/CodeLlama-34b-Instruct-hf` (CodeLlama) |
| Validation     | Run static and runtime checks against the generated project | `meta-llama/CodeLlama-34b-Instruct-hf` (CodeLlama) |
| Diagram        | Generate Mermaid.js architecture and data-flow diagrams | `mistralai/Mixtral-8x7B-Instruct-v0.1` (Mixtral) |
| Chat           | Answer natural-language questions about the paper with RAG-augmented context | `meta-llama/Meta-Llama-3-8B-Instruct` (Llama-3) |

All agent models are configurable via environment variables. Any HuggingFace Inference API-compatible model can be substituted.

---

## License

This project is licensed under the [MIT License](LICENSE).
