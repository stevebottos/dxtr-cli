# Docling PDF to Markdown Converter

Production-grade PDF to Markdown conversion service using Docling, containerized for dependency isolation.

## Project Structure

```
.
├── docker/
│   └── docling/              # Docker build context
│       ├── Dockerfile        # Multi-stage build
│       ├── server.py         # FastAPI service
│       ├── requirements.txt  # Python dependencies
│       └── .dockerignore     # Build exclusions
├── docker-compose.yml        # Service orchestration
├── Makefile                  # Build/deploy commands
├── main.py                   # Standalone batch converter
└── test_docling.py           # Service integration examples
```

## Quick Start

### 1. Start the Service

```bash
make build  # Build Docker image
make up     # Start service (runs on http://localhost:8080)
```

### 2. Test the Service

```bash
make test   # Run test examples
```

### 3. Use in Your Code

```python
import requests
from pathlib import Path

# Convert a PDF
with open('paper.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8080/convert',
        files={'file': f}
    )

result = response.json()
markdown = result['markdown']
Path('output.md').write_text(markdown)
```

## Available Commands

```bash
make build    # Build the Docker image
make up       # Start the service
make down     # Stop the service
make restart  # Restart the service
make logs     # View service logs
make health   # Check service health
make test     # Run test script
make clean    # Remove containers and images
make rebuild  # Clean rebuild and restart
make info     # Show service information
make help     # Show all commands
```

## Service Endpoints

- `POST /convert` - Convert PDF to Markdown
- `GET /health` - Health check
- `GET /` - Service info
- `GET /docs` - Interactive API documentation (Swagger)

## Configuration

Settings optimized for arXiv papers (in `docker/docling/server.py`):

```python
PDF_BACKEND = "pypdfium2"        # Fast, reliable backend
DO_OCR = False                   # No OCR for text-based PDFs
DO_TABLE_STRUCTURE = True        # Preserve table formatting
IMAGES_SCALE = 4.0               # Maximum quality images
IMAGE_EXPORT_FORMAT = "png"      # Lossless image format
```

## Standalone Usage (Without Docker)

For batch converting PDFs without the service:

```bash
python main.py
```

This processes all PDFs in `2025-12-26/` subfolders using the same quality settings.

## Integration in Other Projects

### Copy the Client Class

See `test_docling.py` for a ready-to-use `DoclingClient` class you can copy to your projects:

```python
from test_docling import DoclingClient

client = DoclingClient()
markdown = client.convert_pdf('paper.pdf')
```

### Direct Requests (No Dependencies)

```python
import requests

def convert_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8080/convert',
            files={'file': f},
            timeout=300
        )
    response.raise_for_status()
    return response.json()['markdown']
```

## Why Docker?

This project uses Docker to isolate Docling's dependencies from your main project. This solves dependency conflicts while providing:

- **Clean API** - Simple HTTP interface
- **Zero conflicts** - Complete isolation from your project's dependencies
- **Production-ready** - Service can be deployed anywhere
- **Language agnostic** - Use from any language via HTTP

## Requirements

- Docker and docker-compose
- Python 3.12+ (for standalone `main.py`)
- `requests` library (for client usage)

## Troubleshooting

### Service won't start
```bash
make logs  # Check error messages
```

### Port already in use
Edit `docker-compose.yml` to change port:
```yaml
ports:
  - "8081:8080"  # Use 8081 instead
```

### Conversion timeout
Increase timeout in your client code:
```python
response = requests.post(..., timeout=600)  # 10 minutes
```

## License

MIT
