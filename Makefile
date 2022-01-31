.PHONY : docs lint sync tests

build : lint tests docs

lint :
	flake8 --exclude=docs

tests :
	pytest -v --cov=summaries --cov-fail-under=100 --cov-report=term-missing --cov-report=html

docs :
	rm -rf docs/_build/plot_directive
	sphinx-build . docs/_build

sync : requirements.txt
	pip-sync

requirements.txt : requirements.in setup.py test_requirements.txt
	pip-compile -v -o $@ $<

test_requirements.txt : test_requirements.in setup.py
	pip-compile -v -o $@ $<
