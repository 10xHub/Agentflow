# Makefile for 10xScale Agentflow packaging and publishing

.PHONY: build publish testpublish clean test test-cov docs-serve docs-build docs-deploy

build:
	uv pip install build
	python -m build

publish: build
	uv pip install twine
	twine upload dist/*

testpublish: build
	uv pip install twine
	twine upload --repository testpypi dist/*

clean:
	rm -rf dist build *.egg-info

test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=agentflow --cov-report=html --cov-report=term-missing --cov-report=xml -v



# # 1. publish to PyPI manually (Makefile)
# make clean && make build
# uvx twine check dist/*
# make publish                 # twine upload -> PyPI

# # 2. tag -> GitHub Action builds + cuts the GitHub Release
# git tag v0.8.0               # must be v-prefixed and == pyproject version (0.8.0)
# git push origin v0.8.0
