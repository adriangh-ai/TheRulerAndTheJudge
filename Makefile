.PHONY: test compile verify-source

compile:
	python -m compileall -q src scripts tests

test:
	pytest -q

verify-source:
	python scripts/verify_source_equivalence.py $(ORIGINAL_SOURCE_DIR)
