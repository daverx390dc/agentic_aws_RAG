# RAG Pipeline with LangChain, AWS Bedrock, and OpenSearch

A comprehensive Retrieval-Augmented Generation (RAG) pipeline built with Python, now powered by [LangChain](https://python.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph), featuring AI agents, AWS Bedrock (Claude Sonnet 3 + Titan embeddings), and OpenSearch for vector storage.

## 🚀 Features

- **Modern LangChain Architecture**: Uses LangChain and LangGraph for agent orchestration and RAG
- **AI Agents**: Specialized agents for document processing and query handling, built on LangChain
- **AWS Bedrock Integration**: Uses Claude Sonnet 3 for LLM and Titan for embeddings via LangChain wrappers
- **OpenSearch Vector Database**: High-performance vector storage and similarity search via LangChain
- **Multiple Interfaces**: REST API, CLI, and programmatic access
- **Document Support**: PDF, DOCX, TXT, HTML files
- **Advanced Features**: Query analysis, suggestions, batch processing, health monitoring

## 📁 Project Structure

```
rag_pipeline/
├── agents/                 # AI agents (LangChain-based)
│   ├── document_processor_agent.py
│   └── query_agent.py
├── config/                 # Configuration files
│   ├── settings.py
│   └── env.example
├── data/                   # Sample data and documents
│   └── example_document.txt
├── models/                 # Core pipeline models
│   └── rag_pipeline.py
├── services/               # (Deprecated) Custom service integrations (now handled by LangChain)
│   ├── bedrock_service.py  # Deprecated
│   ├── opensearch_service.py # Deprecated
│   └── api_service.py
├── utils/                  # Utility functions
│   ├── logger.py
│   └── text_processing.py
├── scripts/                # CLI and scripts
│   └── cli.py
├── tests/                  # Test files
├── docs/                   # Documentation
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build file
├── docker-compose.yml      # Docker Compose orchestration
├── .dockerignore           # Docker ignore file
└── README.md              # This file
```

> **Note:** `services/bedrock_service.py` and `services/opensearch_service.py` are now deprecated. All Bedrock and OpenSearch logic is handled via LangChain's wrappers.

## 🛠️ Installation

### Prerequisites

1. **Python 3.8+**
2. **OpenSearch** (local or cloud instance)
3. **AWS Account** with Bedrock access
4. **Docker** (optional, for OpenSearch or full containerization)

### Setup (Local)

1. **Clone and navigate to the project:**
   ```bash
   cd rag_pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp config/env.example .env
   # Edit .env with your AWS credentials and OpenSearch settings
   ```

4. **Start OpenSearch** (if using local instance):
   ```bash
   # Using Docker
   docker run -d \
     --name opensearch \
     -p 9200:9200 \
     -p 9600:9600 \
     -e "discovery.type=single-node" \
     -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
     opensearchproject/opensearch:latest
   ```

## 🐳 Docker & Containerization

This project is fully containerized for easy deployment and development.

### Build and Run with Docker Compose

1. **Copy and configure your environment variables:**
   ```bash
   cp config/env.example .env
   # Edit .env as needed
   ```

2. **Build and start the stack:**
   ```bash
   docker-compose up --build
   ```
   This will start both the RAG pipeline app and an OpenSearch instance.

3. **Access the services:**
   - RAG API: [http://localhost:8000](http://localhost:8000)
   - OpenSearch: [http://localhost:9200](http://localhost:9200)

4. **Run CLI commands inside the container:**
   ```bash
   docker-compose exec rag-pipeline python scripts/cli.py health
   docker-compose exec rag-pipeline python scripts/cli.py ingest data/example_document.txt
   docker-compose exec rag-pipeline python scripts/cli.py query "What is machine learning?"
   ```

5. **Stop and remove containers:**
   ```bash
   docker-compose down
   ```

### Dockerfile
- The `Dockerfile` builds the app image, installs all dependencies, and sets up the default entrypoint to run `main.py`.

### .dockerignore
- The `.dockerignore` file ensures that unnecessary files (e.g., `.git`, `__pycache__`, `.env`, logs) are not copied into the Docker image.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# OpenSearch Configuration
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin
OPENSEARCH_INDEX_NAME=rag_documents
OPENSEARCH_VECTOR_DIMENSION=1536

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4096
TEMPERATURE=0.1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

## 🚀 Usage

### 1. Command Line Interface (CLI)

#### Health Check
```bash
python scripts/cli.py health
# Or inside Docker:
docker-compose exec rag-pipeline python scripts/cli.py health
```

#### Ingest Documents
```bash
# Single file
python scripts/cli.py ingest data/example_document.txt
# Or inside Docker:
docker-compose exec rag-pipeline python scripts/cli.py ingest data/example_document.txt

# Multiple files
python scripts/cli.py ingest file1.txt file2.pdf file3.docx
# Or inside Docker:
docker-compose exec rag-pipeline python scripts/cli.py ingest file1.txt file2.pdf file3.docx

# Directory
python scripts/cli.py ingest-dir /path/to/documents/
# Or inside Docker:
docker-compose exec rag-pipeline python scripts/cli.py ingest-dir /app/data/

# With custom source names
python scripts/cli.py ingest file1.txt file2.txt -s "source1" -s "source2"
```

#### Query the Pipeline
```bash
# Simple query with default (RAG) agent
python scripts/cli.py query "What is machine learning?"
# Or inside Docker:
docker-compose exec rag-pipeline python scripts/cli.py query "What is machine learning?"

# Use the ReAct agent for advanced reasoning and tool use
python scripts/cli.py query "What is machine learning?" --agent-type react
# Or inside Docker:
docker-compose exec rag-pipeline python scripts/cli.py query "What is machine learning?" --agent-type react

# With options
python scripts/cli.py query "Explain neural networks" --top-k 10 --include-sources --agent-type rag

# Batch queries
python scripts/cli.py batch-query "What is AI?" "How does ML work?" "Explain deep learning"

# From file
python scripts/cli.py batch-query --input-file questions.txt
```

### Agent Types

- **RAG agent (default):** Uses retrieval-augmented generation for question answering.
- **ReAct agent:** Uses LangChain's ReAct agent type, combining reasoning and tool-calling (search and summarization) for more complex, multi-step queries.

To use the ReAct agent, add `--agent-type react` to your CLI query command.

#### Get Suggestions
```bash
python scripts/cli.py suggestions "machine learning"
```

#### Analyze Query
```bash
python scripts/cli.py analyze "What is the difference between supervised and unsupervised learning?"
```

#### Pipeline Statistics
```bash
python scripts/cli.py stats
```

### 2. REST API

#### Start the API Server
```bash
python services/api_service.py
# Or inside Docker (if you change the entrypoint):
docker-compose exec rag-pipeline python services/api_service.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /stats` - Pipeline statistics
- `POST /query` - Query the pipeline
- `POST /query/batch` - Batch queries
- `POST /query/with-context` - Query with context
- `GET /suggestions` - Get query suggestions
- `POST /analyze-query` - Analyze query intent
- `POST /ingest/files` - Ingest multiple files
- `POST /ingest/upload` - Upload and ingest file
- `POST /ingest/directory` - Ingest directory
- `DELETE /sources/{source_name}` - Remove source
- `POST /reset` - Reset pipeline

#### Example API Usage

```bash
# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 5}'

# Ingest files
curl -X POST "http://localhost:8000/ingest/files" \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["data/example_document.txt"]}'

# Health check
curl "http://localhost:8000/health"
```

### 3. Programmatic Usage

```python
from models.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents
results = pipeline.ingest_documents(["data/example_document.txt"])

# Query
response = pipeline.query("What is artificial intelligence?")
print(response["response"])

# Batch query
questions = ["What is ML?", "How does AI work?", "Explain neural networks"]
results = pipeline.batch_query(questions)

# Get suggestions
suggestions = pipeline.get_query_suggestions("machine learning")

# Analyze query
analysis = pipeline.analyze_query("What is the difference between AI and ML?")

# Health check
health = pipeline.health_check()
```

## 🔧 AI Agents (LangChain-based)

### Document Processor Agent
- Handles document ingestion and preprocessing using LangChain document loaders and splitters
- Supports multiple file formats (PDF, DOCX, TXT, HTML)
- Generates embeddings using LangChain's BedrockEmbeddings
- Indexes documents in OpenSearch via LangChain's OpenSearchVectorSearch

### Query Agent
- Processes user queries using LangChain's RetrievalQA and agent framework
- Performs vector similarity search via LangChain
- Generates responses using Claude Sonnet 3 (via Bedrock)
- Provides query analysis and suggestions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │   Documents     │    │   API/CLI       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌───────────────────────────────┐
│   LangChain Agents & Chains  │
│  (QueryAgent, DocProcessor)  │
└─────────┬─────────────────────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐
│ AWS Bedrock     │    │   OpenSearch    │
│ Claude Sonnet 3 │    │ Vector Database │
│ Titan Embedding │    └─────────────────┘
└─────────────────┘
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
# Or inside Docker:
docker-compose exec rag-pipeline pytest tests/
```

## 📊 Monitoring

### Health Check
```bash
python scripts/cli.py health
# Or inside Docker:
docker-compose exec rag-pipeline python scripts/cli.py health
```

### Statistics
```bash
python scripts/cli.py stats
# Or inside Docker:
docker-compose exec rag-pipeline python scripts/cli.py stats
```

### Logs
Logs are stored in `logs/rag_pipeline.log` with rotation and retention policies.

## 🔒 Security

- Use environment variables for sensitive credentials
- Implement proper authentication for production deployments
- Use HTTPS for API endpoints in production
- Regularly rotate AWS credentials

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production
1. Set up proper environment variables
2. Use a production WSGI server (e.g., Gunicorn)
3. Set up reverse proxy (e.g., Nginx)
4. Configure monitoring and logging
5. Set up backup and recovery procedures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## 🔄 Roadmap

- [ ] Web UI interface
- [ ] Advanced query processing
- [ ] Multi-language support
- [ ] Real-time document updates
- [ ] Advanced analytics dashboard
- [ ] Integration with more LLM providers
- [ ] Enhanced security features 