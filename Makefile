setup:
	python3 -m venv myPythonTemplate

install:
	. myPythonTemplate/bin/activate; \
	pip install -r requirements.txt

run:
	. myPythonTemplate/bin/activate; \
	python src/main.py

test:
	. myPythonTemplate/bin/activate; \
	pytest tests/

.PHONY: setup install run test
