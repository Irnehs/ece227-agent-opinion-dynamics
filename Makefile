NUM_AGENTS ?= 20
MODEL_NAME=llama3.2:1b

PHONY: build_agents
build_agents:
	sh ./build_agents.sh $(NUM_AGENTS)

PHONY: up
up:
	echo "Starting agent-1 to provision shared manifest"
	docker compose up -d agent-1
	echo "Pulling $(MODEL_NAME)..."
	docker compose exec agent-1 ollama pull $(MODEL_NAME)
	echo "Download complete. Staring remaining agents"
	docker compose up -d

PHONY: down
down:
	docker compose down

PHONY: clean
clean: down
	rm docker-compose.yml


