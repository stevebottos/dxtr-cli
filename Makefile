.PHONY: build up down restart logs health clean test process-papers

# Build the Docker image
build:
	docker compose build

# Start the service
up:
	docker compose up -d

# Stop the service
down:
	docker compose down

# Restart the service
restart: down up

# View logs
logs:
	docker compose logs -f docling

# Check service health
health:
	@curl -s http://localhost:8080/health | python -m json.tool || echo "Service not responding"

# Clean up containers and images
clean:
	docker compose down -v
	docker rmi docling-test-docling 2>/dev/null || true

# Rebuild and restart
rebuild: clean build up

# Run test script
test:
	python test_docling.py

# Process daily papers (convert PDFs to markdown)
process-papers:
	python process_daily_papers.py $(DATE)

# Show service info
info:
	@echo "Docling PDF Converter Service"
	@echo "=============================="
	@echo "Service URL: http://localhost:8080"
	@echo "Health check: make health"
	@echo "View logs: make logs"
	@echo "API docs: http://localhost:8080/docs"
	@echo ""
	@docker compose ps

help:
	@echo "Available commands:"
	@echo "  make build           - Build the Docker image"
	@echo "  make up              - Start the service (detached)"
	@echo "  make down            - Stop the service"
	@echo "  make restart         - Restart the service"
	@echo "  make logs            - View service logs"
	@echo "  make health          - Check service health"
	@echo "  make test            - Run test script"
	@echo "  make process-papers  - Process daily papers (convert PDFs to markdown)"
	@echo "                         Usage: make process-papers [DATE=2025-12-26]"
	@echo "  make clean           - Remove containers and images"
	@echo "  make rebuild         - Clean, rebuild, and restart"
	@echo "  make info            - Show service information"
