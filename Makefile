.PHONY: litellm agent-server mock-conversation dev stop

# Start LiteLLM proxy (port 4000)
litellm:
	litellm --config dxtr/litellm_config.yaml --port 4000

# Start agent server (port 8000)
agent-server:
	cd dxtr && python server.py

# Run mock conversation (requires both servers running)
mock-conversation:
	python mock_conversation.py

# Start both services (LiteLLM in background, agent server in foreground)
dev:
	@echo "Starting LiteLLM proxy on :4000..."
	@litellm --config dxtr/litellm_config.yaml --port 4000 & echo $$! > .litellm.pid
	@sleep 2
	@echo "Starting agent server on :8000..."
	@cd dxtr && python server.py || (cd .. && make stop && exit 1)

# Stop background services
stop:
	@if [ -f .litellm.pid ]; then \
		kill $$(cat .litellm.pid) 2>/dev/null || true; \
		rm .litellm.pid; \
		echo "Stopped LiteLLM proxy"; \
	fi
	@pkill -f "python server.py" 2>/dev/null || true
	@echo "Stopped agent server"
