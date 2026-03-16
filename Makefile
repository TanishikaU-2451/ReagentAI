# ReagentAI Makefile - Automation Commands
# Usage: make <command>

.PHONY: help install dev test lint format docker clean setup

# Default target
help: ## Show this help message
	@echo "ReagentAI Development Commands"
	@echo "=============================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup and Installation
setup: ## Complete development environment setup
	@echo "🚀 Setting up ReagentAI development environment..."
	chmod +x setup-dev.sh
	./setup-dev.sh

install: ## Install all dependencies
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install black flake8 mypy pytest pytest-cov pre-commit
	cd frontend/nextjs_app && npm install

# Development
dev: ## Start development servers (backend + frontend)
	@echo "🔥 Starting development servers..."
	@trap 'kill %1; kill %2' INT; \
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload & \
	cd frontend/nextjs_app && npm run dev &\
	wait

dev-backend: ## Start only backend server
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

dev-frontend: ## Start only frontend server
	cd frontend/nextjs_app && npm run dev

# Testing
test: ## Run all tests with coverage
	pytest backend/ -v --cov=backend --cov-report=html --cov-report=xml
	cd frontend/nextjs_app && npm test -- --passWithNoTests

test-backend: ## Run only backend tests
	pytest backend/ -v --cov=backend

test-frontend: ## Run only frontend tests
	cd frontend/nextjs_app && npm test -- --passWithNoTests

test-watch: ## Run tests in watch mode
	pytest backend/ -v --cov=backend -f &
	cd frontend/nextjs_app && npm run test:watch

# Code Quality
lint: ## Run all linters
	flake8 backend/
	mypy backend/ --ignore-missing-imports
	cd frontend/nextjs_app && npm run lint

format: ## Format all code
	black backend/
	isort backend/
	cd frontend/nextjs_app && npm run format

quality: format lint test ## Run all quality checks (format + lint + test)

# Git and Pre-commit
hooks: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

hooks-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

hooks-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Git workflow
commit: ## Interactive conventional commit
	cz commit

bump: ## Bump version and create release
	cz bump

# Docker
docker: ## Build and run with Docker Compose
	docker-compose up --build

docker-build: ## Build Docker images
	docker-compose build

docker-clean: ## Clean Docker containers and images
	docker-compose down -v --rmi all

# Deployment
deploy-backend: ## Deploy backend to Railway
	railway up --service backend

deploy-frontend: ## Deploy frontend to Vercel
	cd frontend/nextjs_app && vercel --prod

# Utilities
clean: ## Clean build artifacts and cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache
	cd frontend/nextjs_app && rm -rf .next node_modules/.cache

logs: ## View application logs
	tail -f logs/reagentai.log

status: ## Check system status
	@echo "🔍 ReagentAI System Status"
	@echo "=========================="
	@echo "Python version: $(shell python --version 2>/dev/null || echo 'Not installed')"
	@echo "Node version: $(shell node --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker version: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Git status:"
	@git status --porcelain || echo "Not a git repository"
	@echo "Backend running: $(shell curl -s http://localhost:8000/health >/dev/null && echo 'Yes' || echo 'No')"
	@echo "Frontend running: $(shell curl -s http://localhost:3000 >/dev/null && echo 'Yes' || echo 'No')"

# CI/CD simulation
ci: ## Simulate CI/CD pipeline locally
	@echo "🤖 Running CI/CD simulation..."
	make format
	make lint
	make test
	@echo "✅ CI/CD simulation passed!"

# Quick commands for daily workflow
start: dev ## Alias for dev
stop: ## Stop all development servers
	@pkill -f "uvicorn backend.main" || true
	@pkill -f "next-server" || true
	@pkill -f "next dev" || true
	@echo "🛑 Development servers stopped"

restart: stop start ## Restart development servers

# Development utilities
shell: ## Start Python shell with project context
	cd backend && python -c "import sys; sys.path.insert(0, '..'); from backend.main import app; import backend"

notebook: ## Start Jupyter notebook (if installed)
	jupyter notebook --notebook-dir=. --ip=0.0.0.0

# Database and storage
reset-vector-store: ## Reset FAISS vector store
	rm -rf vector_store/*.faiss vector_store/*.pkl
	@echo "🗑️ Vector store reset"

reset-logs: ## Clear all log files
	rm -f logs/*.log
	@echo "🗑️ Logs cleared"

reset-projects: ## Clear generated projects
	rm -rf generated_projects/*
	@echo "🗑️ Generated projects cleared"

reset-all: reset-vector-store reset-logs reset-projects ## Reset all data
	@echo "🗑️ All data reset"