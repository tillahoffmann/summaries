.PHONY : docs lint sync tests figures ${BENCHMARK_ROOT} ${BENCHMARK_DATA_ROOT} \
	${BENCHMARK_SAMPLE_ROOT} ${COAL_ROOT} ${COAL_DATA_ROOT} ${COAL_SAMPLE_ROOT}

ENV = NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

build : lint tests docs

lint :
	flake8 --exclude=docs

tests :
	pytest -v --cov=summaries --cov-fail-under=100 --cov-report=term-missing --cov-report=html \
		--durations=10

docs :
	rm -rf docs/_build/plot_directive
	sphinx-build . docs/_build

sync : requirements.txt
	pip-sync

requirements.txt : requirements.in setup.py test_requirements.txt
	pip-compile -v -o $@ $<

test_requirements.txt : test_requirements.in setup.py
	pip-compile -v -o $@ $<

# Generate figures =================================================================================

FIGURES = bimodal broad_posterior piecewise_likelihood benchmark
FIGURE_TARGETS = $(addprefix figures/,${FIGURES:=.pdf})
figures : ${FIGURE_TARGETS}

# Dictionary of seeds for visualization purposes.
FIGURE_SEED_bimodal = 0
FIGURE_SEED_broad_posterior = 0
FIGURE_SEED_piecewise_likelihood = 0
FIGURE_SEED_benchmark = 3

# Dictionary of methods for plotting.
FIGURE_FUNC_bimodal = summaries.examples.bimodal:_plot_example
FIGURE_FUNC_broad_posterior = summaries.examples.broad_posterior:_plot_example
FIGURE_FUNC_piecewise_likelihood = summaries.examples.piecewise_likelihood:_plot_example
FIGURE_FUNC_benchmark = summaries.benchmark:_plot_example

${FIGURE_TARGETS} : figures/%.pdf : scrartcl.mplstyle
	SEED=${FIGURE_SEED_$*} python -m summaries.scripts.plot --style=$< ${FIGURE_FUNC_$*} $@
