.PHONY : docs lint sync tests figures

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

FIGURES = bimodal broad_posterior piecewise_likelihood benchmark
FIGURE_TARGETS = $(addprefix figures/,${FIGURES:=.pdf})
figures : ${FIGURE_TARGETS}

# Dictionary of seeds for visualization purposes.
SEED_bimodal = 0
SEED_broad_posterior = 0
SEED_piecewise_likelihood = 0
SEED_benchmark = 3

${FIGURE_TARGETS} : figures/%.pdf : summaries/examples/%.py
	python scripts/plot.py --seed=${SEED_$*} --style=scrartcl.mplstyle summaries.examples.$*:_plot_example $@
