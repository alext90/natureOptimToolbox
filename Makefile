setup:
	python3 -m venv natureOptimToolbox

install:
	. natureOptimToolbox/bin/activate; \
	pip install -r requirements.txt

run:
	. natureOptimToolbox/bin/activate; \
	python src/main.py

test:
	. natureOptimToolbox/bin/activate; \
	pytest tests/

.PHONY: setup install run test
