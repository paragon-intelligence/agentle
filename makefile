.PHONY: release docs

release:
	uv run release.py

docs:
	@echo "Gerando documentação..."
	cd docs && make html
