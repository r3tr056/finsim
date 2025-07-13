.PHONY: help build push deploy-local deploy-prod test clean

# Default target
help: ## Show this help message
	@echo "FinSim - Production-grade Financial Simulation Platform"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment variables
DOCKER_REGISTRY ?= docker.io
IMAGE_PREFIX ?= finsim
VERSION ?= latest
NAMESPACE ?= finsim

# Service list
SERVICES = market-data simulation-engine agents risk-analytics portfolio auth-service
FRONTEND = dashboard

# Build all Docker images
build: ## Build all Docker images
	@echo "Building all FinSim services..."
	@for service in $(SERVICES); do \
		echo "Building $$service..."; \
		docker build -t $(DOCKER_REGISTRY)/$(IMAGE_PREFIX)/$$service:$(VERSION) ./services/$$service/; \
	done
	@echo "Building dashboard..."
	@docker build -t $(DOCKER_REGISTRY)/$(IMAGE_PREFIX)/dashboard:$(VERSION) ./ui/dashboard/

# Push all Docker images to registry
push: build ## Push all Docker images to registry
	@echo "Pushing all FinSim images to registry..."
	@for service in $(SERVICES); do \
		echo "Pushing $$service..."; \
		docker push $(DOCKER_REGISTRY)/$(IMAGE_PREFIX)/$$service:$(VERSION); \
	done
	@echo "Pushing dashboard..."
	@docker push $(DOCKER_REGISTRY)/$(IMAGE_PREFIX)/dashboard:$(VERSION)

# Install dependencies
install-deps: ## Install development dependencies
	@echo "Installing Python dependencies..."
	@for service in $(SERVICES); do \
		echo "Installing dependencies for $$service..."; \
		cd services/$$service && pip install -r requirements.txt && cd ../..; \
	done
	@echo "Installing Node.js dependencies..."
	@cd ui/dashboard && npm install

# Run tests
test: ## Run all tests
	@echo "Running Python tests..."
	@for service in $(SERVICES); do \
		echo "Testing $$service..."; \
		cd services/$$service && python -m pytest tests/ --cov=app && cd ../..; \
	done
	@echo "Running frontend tests..."
	@cd ui/dashboard && npm test

# Lint code
lint: ## Lint all code
	@echo "Linting Python code..."
	@for service in $(SERVICES); do \
		echo "Linting $$service..."; \
		cd services/$$service && flake8 app && black --check app && mypy app --ignore-missing-imports && cd ../..; \
	done
	@echo "Linting frontend code..."
	@cd ui/dashboard && npm run lint && npm run type-check

# Format code
format: ## Format all code
	@echo "Formatting Python code..."
	@for service in $(SERVICES); do \
		echo "Formatting $$service..."; \
		cd services/$$service && black app && cd ../..; \
	done
	@echo "Formatting frontend code..."
	@cd ui/dashboard && npm run format

# Local development deployment with Docker Compose
deploy-local: ## Deploy FinSim locally using Docker Compose
	@echo "Deploying FinSim locally..."
	@docker compose up -d --build
	@echo ""
	@echo "🚀 FinSim is starting up..."
	@echo ""
	@echo "Services will be available at:"
	@echo "  Dashboard:        http://localhost:3000"
	@echo "  API Gateway:      http://localhost:8000"
	@echo "  Market Data:      http://localhost:8001"
	@echo "  Simulation:       http://localhost:8002"
	@echo "  Agents:           http://localhost:8003"
	@echo "  Risk Analytics:   http://localhost:8004"
	@echo "  Portfolio:        http://localhost:8005"
	@echo "  Auth Service:     http://localhost:8006"
	@echo "  Grafana:          http://localhost:3001 (admin/admin)"
	@echo "  Prometheus:       http://localhost:9090"
	@echo ""
	@echo "Wait a few minutes for all services to start up completely."

# Stop local deployment
stop-local: ## Stop local Docker Compose deployment
	@echo "Stopping FinSim local deployment..."
	@docker compose down

# Production deployment to Kubernetes
deploy-prod: ## Deploy FinSim to production Kubernetes cluster
	@echo "Deploying FinSim to production..."
	@helm upgrade --install $(NAMESPACE) ./helm/finsim \
		--namespace $(NAMESPACE) \
		--create-namespace \
		--set global.environment=production \
		--set image.registry=$(DOCKER_REGISTRY) \
		--set image.tag=$(VERSION) \
		--wait --timeout=15m
	@echo ""
	@echo "🚀 FinSim deployed to production!"
	@echo ""
	@kubectl get pods -n $(NAMESPACE)
	@echo ""
	@echo "To access the dashboard:"
	@kubectl get ingress -n $(NAMESPACE)

# Staging deployment
deploy-staging: ## Deploy FinSim to staging environment
	@echo "Deploying FinSim to staging..."
	@helm upgrade --install $(NAMESPACE)-staging ./helm/finsim \
		--namespace $(NAMESPACE)-staging \
		--create-namespace \
		--set global.environment=staging \
		--set image.registry=$(DOCKER_REGISTRY) \
		--set image.tag=$(VERSION) \
		--wait --timeout=10m

# Uninstall from Kubernetes
undeploy: ## Remove FinSim from Kubernetes
	@echo "Removing FinSim deployment..."
	@helm uninstall $(NAMESPACE) --namespace $(NAMESPACE) || true
	@kubectl delete namespace $(NAMESPACE) || true

# Clean up Docker resources
clean: ## Clean up Docker images and containers
	@echo "Cleaning up Docker resources..."
	@docker compose down -v --remove-orphans || true
	@docker system prune -f
	@for service in $(SERVICES) $(FRONTEND); do \
		docker rmi $(DOCKER_REGISTRY)/$(IMAGE_PREFIX)/$$service:$(VERSION) 2>/dev/null || true; \
	done

# Development helpers
dev-backend: ## Start only backend services for development
	@echo "Starting backend services..."
	@docker compose up -d postgres redis kafka zookeeper influxdb
	@echo "Backend infrastructure started. Run individual services manually for development."

dev-frontend: ## Start frontend in development mode
	@echo "Starting frontend development server..."
	@cd ui/dashboard && npm run dev

# Database operations
db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	@docker compose exec postgres psql -U finsim -d finsim -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "Resetting database..."
	@docker compose down postgres
	@docker volume rm finsim_postgres_data 2>/dev/null || true
	@docker compose up -d postgres
	@sleep 10
	@make db-migrate

# Monitoring and logs
logs: ## Show logs from all services
	@docker compose logs -f

logs-service: ## Show logs from specific service (usage: make logs-service SERVICE=market-data)
	@docker compose logs -f $(SERVICE)

monitor: ## Open monitoring dashboards
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3001 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:3001 http://localhost:9090; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:3001; xdg-open http://localhost:9090; \
	fi

# Load testing
load-test: ## Run load tests against the deployed system
	@echo "Running load tests..."
	@k6 run tests/load/market-data-load.js
	@k6 run tests/load/trading-load.js

# Backup and restore
backup: ## Create backup of the system
	@echo "Creating system backup..."
	@kubectl exec -n $(NAMESPACE) deployment/postgres -- pg_dump -U finsim finsim > backup-$(shell date +%Y%m%d-%H%M%S).sql

# Health check
health: ## Check health of all services
	@echo "Checking service health..."
	@curl -s http://localhost:8001/health | jq . || echo "Market Data service not responding"
	@curl -s http://localhost:8002/health | jq . || echo "Simulation Engine service not responding"
	@curl -s http://localhost:8003/health | jq . || echo "Agents service not responding"
	@curl -s http://localhost:8004/health | jq . || echo "Risk Analytics service not responding"
	@curl -s http://localhost:8005/health | jq . || echo "Portfolio service not responding"
	@curl -s http://localhost:8006/health | jq . || echo "Auth service not responding"

# Demo data
demo-data: ## Load demo data for testing
	@echo "Loading demo data..."
	@python scripts/load_demo_data.py

# Documentation
docs: ## Generate and serve documentation
	@echo "Generating documentation..."
	@cd docs && mkdocs serve

# Security scan
security-scan: ## Run security scans on Docker images
	@echo "Running security scans..."
	@for service in $(SERVICES) $(FRONTEND); do \
		echo "Scanning $$service..."; \
		trivy image $(DOCKER_REGISTRY)/$(IMAGE_PREFIX)/$$service:$(VERSION); \
	done

# Performance test
perf-test: ## Run performance tests
	@echo "Running performance tests..."
	@python tests/performance/benchmark.py

# Full CI/CD pipeline locally
ci: lint test build ## Run full CI pipeline locally

# Setup development environment
setup-dev: ## Setup development environment
	@echo "Setting up development environment..."
	@make install-deps
	@make deploy-local
	@echo ""
	@echo "✅ Development environment ready!"
	@echo "Dashboard: http://localhost:3000"
	@echo "Login with: admin@finsim.com / admin123"