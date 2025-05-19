.PHONY: release docs clear-cache

release:
	uv run release.py

docs:
	@echo "Gerando documentação..."
	cd docs && make html

clear-cache:
	@echo "Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "Removing mypy cache..."
	rm -rf .mypy_cache
