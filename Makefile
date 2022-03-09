.SECONDEXPANSION :
.PHONY : docs lint sync tests figures ${BENCHMARK_ROOT} ${BENCHMARK_DATA_ROOT} \
	${BENCHMARK_SAMPLE_ROOT}

ENV = NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

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
	python -m summaries.scripts.plot --seed=${FIGURE_SEED_$*} --style=$< \
		${FIGURE_FUNC_$*} $@

# Generate benchmark data and run inference ========================================================

# Path for everything benchmark-related.
BENCHMARK_ROOT = workspace/benchmark
# Dataset to evaluate on.
MODE ?= test
# Dataset to use as the reference table.
REFERENCE ?= train

# Generate benchmark data --------------------------------------------------------------------------

BENCHMARK_DATA_ROOT = ${BENCHMARK_ROOT}/data
BENCHMARK_DATA_NAMES = train validation test debug
BENCHMARK_DATA_TARGETS = $(addprefix ${BENCHMARK_DATA_ROOT}/,${BENCHMARK_DATA_NAMES:=.pkl})

BENCHMARK_DATA_SIZE_train = 1000000
BENCHMARK_DATA_SIZE_validation = 10000
BENCHMARK_DATA_SIZE_test = 1000
BENCHMARK_DATA_SIZE_debug = 100

BENCHMARK_DATA_SEED_train = 0
BENCHMARK_DATA_SEED_validation = 1
BENCHMARK_DATA_SEED_test = 2
BENCHMARK_DATA_SEED_debug = 3

${BENCHMARK_DATA_ROOT} : ${BENCHMARK_DATA_TARGETS}

${BENCHMARK_DATA_TARGETS} : ${BENCHMARK_DATA_ROOT}/%.pkl :
	${ENV} python -m summaries.scripts.generate_benchmark_data --seed=${BENCHMARK_DATA_SEED_$*} \
		${BENCHMARK_DATA_SIZE_$*} $@

${BENCHMARK_ROOT}/generate_benchmark_data.prof :
	${ENV} python -m cProfile -o $@ -m summaries.scripts.generate_benchmark_data --seed=0 100000 \
		${BENCHMARK_DATA_ROOT}/temp.pkl

# Train a mixture density network ------------------------------------------------------------------

BENCHMARK_MDN = ${BENCHMARK_ROOT}/${REFERENCE}_mdn.pt
BENCHMARK_MDN_COMPRESSOR = ${BENCHMARK_ROOT}/${REFERENCE}_mdn_compressor.pt

${BENCHMARK_ROOT}/mdn : ${BENCHMARK_MDN} ${BENCHMARK_MDN_COMPRESSOR}

${BENCHMARK_MDN} ${BENCHMARK_MDN_COMPRESSOR} : ${BENCHMARK_DATA_ROOT}/${REFERENCE}.pkl \
		${BENCHMARK_DATA_ROOT}/validation.pkl
	${ENV} python -m summaries.scripts.train_benchmark_mdn $^ ${BENCHMARK_MDN} \
		${BENCHMARK_MDN_COMPRESSOR}

# Draw posterior samples using different algorithms ------------------------------------------------

BENCHMARK_SAMPLE_ROOT = ${BENCHMARK_ROOT}/samples
# Number of posterior samples.
NUM_SAMPLES ?= 5000
# Algorithms, additional algorithm options, and additional algorithm dependencies.
ALGORITHMS = $(shell python -m summaries.scripts.run_inference --list)
ALGORITHM_OPTIONS_stan = '--sample_options={"keep_fits": true, "seed": 0, "adapt_delta": 0.99}'
ALGORITHM_OPTIONS_mdn = "--cls_options={\"path\": \"${BENCHMARK_MDN}\"}"
ALGORITHM_OPTIONS_mdn_compressor = "--cls_options={\"path\": \"${BENCHMARK_MDN_COMPRESSOR}\"}"
ALGORITHM_DEPS_mdn = ${BENCHMARK_MDN}
ALGORITHM_DEPS_mdn_compressor = ${BENCHMARK_MDN_COMPRESSOR}

BENCHMARK_SAMPLE_TARGETS = $(addprefix ${BENCHMARK_SAMPLE_ROOT}/,${ALGORITHMS:=.pkl})

${BENCHMARK_SAMPLE_ROOT} : ${BENCHMARK_SAMPLE_TARGETS}

${BENCHMARK_SAMPLE_TARGETS} : ${BENCHMARK_SAMPLE_ROOT}/%.pkl : \
		${BENCHMARK_DATA_ROOT}/${REFERENCE}.pkl ${BENCHMARK_DATA_ROOT}/${MODE}.pkl \
		$${ALGORITHM_DEPS_$$*}
# Args: custom options for the algo, name of algo, train and test data, # samples, output path.
	${ENV} python -m summaries.scripts.run_inference ${ALGORITHM_OPTIONS_$*} $* \
		${BENCHMARK_DATA_ROOT}/${REFERENCE}.pkl ${BENCHMARK_DATA_ROOT}/${MODE}.pkl ${NUM_SAMPLES} $@
