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

# Generate figures ---------------------------------------------------------------------------------

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

# Generate benchmark data --------------------------------------------------------------------------

BENCHMARK_NAMES = train validation test
BENCHMARK_TARGETS = $(addprefix workspace/,${BENCHMARK_NAMES:=.pkl})

BENCHMARK_SIZE_train = 1000000
BENCHMARK_SIZE_validation = 10000
BENCHMARK_SIZE_test = 10000

BENCHMARK_SEED_train = 0
BENCHMARK_SEED_validation = 1
BENCHMARK_SEED_test = 2

benchmark_data : ${BENCHMARK_TARGETS}

${BENCHMARK_TARGETS} : workspace/%.pkl : summaries/examples/benchmark.py
	generate_benchmark_data --seed=${BENCHMARK_SEED_$*} ${BENCHMARK_SIZE_$*} $@
