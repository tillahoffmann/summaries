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
	python -m summaries.scripts.plot --seed=${FIGURE_SEED_$*} --style=$< \
		${FIGURE_FUNC_$*} $@

# Generate benchmark data --------------------------------------------------------------------------

BENCHMARK_NAMES = train validation test debug
BENCHMARK_TARGETS = $(addprefix workspace/,${BENCHMARK_NAMES:=.pkl})

BENCHMARK_SIZE_train = 1000000
BENCHMARK_SIZE_validation = 10000
BENCHMARK_SIZE_test = 1000
BENCHMARK_SIZE_debug = 100

BENCHMARK_SEED_train = 0
BENCHMARK_SEED_validation = 1
BENCHMARK_SEED_test = 2
BENCHMARK_SEED_debug = 3

benchmark_data : ${BENCHMARK_TARGETS}

${BENCHMARK_TARGETS} : workspace/%.pkl : summaries/scripts/generate_benchmark_data.py summaries/benchmark.py
	python -m summaries.scripts.generate_benchmark_data --seed=${BENCHMARK_SEED_$*} ${BENCHMARK_SIZE_$*} $@

# Run inference on benchmark data ------------------------------------------------------------------

ALGORITHMS = naive
# Dataset to evaluate on.
MODE ?= test
# Dataset to use as the reference table.
REFERENCE ?= train
INFERENCE_TARGETS = $(addprefix workspace/${MODE}_,${ALGORITHMS:=.pkl})
NUM_SAMPLES = 113

inference : ${INFERENCE_TARGETS}
${INFERENCE_TARGETS} : workspace/${MODE}_%.pkl : workspace/${REFERENCE}.pkl workspace/${MODE}.pkl
	python -m summaries.scripts.run_inference $* workspace/${REFERENCE}.pkl workspace/${MODE}.pkl \
		${NUM_SAMPLES} $@
