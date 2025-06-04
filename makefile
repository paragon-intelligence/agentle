.PHONY: release docs clear-cache evolution evolution-down

release:
	uv run release.py

docs:
	@echo "Gerando documentação..."
	cd docs && make html

evolution:
	cd docker && docker compose up -d

evolution-down:
	cd docker && docker compose down

clear-cache:
	@echo "Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Removing mypy cache..."
	rm -rf .mypy_cache
