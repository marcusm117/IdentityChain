#########
# BUILD #
#########
develop:  ## install dependencies and the library
	python -m pip install -e .[develop]

build:  ## build the python library
	python setup.py build build_ext --inplace

install:  ## install library
	python -m pip install .


#########
# LINTS #
#########
lint:  ## run static analysis with black, flake8, and pylint, 	python -m pylint --disable=C0301,R1720,C0114,C0103,W1114 identitychain
	python -m black --check identitychain scripts examples
	python -m flake8 --max-line-length=150 --extend-ignore=E203 identitychain scripts examples

# Alias
lints: lint

format:  ## run autoformatting with black
	python -m black identitychain/ scripts/ examples/

# Alias
fix: format

check:  ## check assets for packaging
	check-manifest -v

# Alias
checks: check

annotate:  ## run type checking
	python -m mypy ./identitychain ./scripts

# Alias
type: annotate


#########
# TESTS #
#########
test:  ## clean and run unit tests, python -m pytest -vv identitychain/tests
	bash scripts/run_tests.sh

# Alias
tests: test

coverage:  ## clean and run unit tests with coverage
	python -m pytest -v identitychain/tests --cov=identitychain --cov-branch --cov-fail-under=100 --junitxml=python_junit.xml --cov-report term-missing

# Alias
cov: coverage


###########
# VERSION #
###########
show-version:  ## doesn't work on Windows
	bump2version --dry-run --allow-dirty setup.py --list | grep current | awk -F= '{print $2}'

patch:
	bump2version patch

minor:
	bump2version minor

major:
	bump2version major


########
# DIST #
########
dist-build:  ## build python dist, can also add bdist_wheel
	python setup.py sdist

dist-check:
	python -m twine check dist/*

dist: deep-clean check dist-build dist-check  ## build dists

publish:  ## upload python assets
	echo "would usually run python -m twine upload dist/* --skip-existing"

# Alias
pub: publish


#########
# CLEAN #
#########
deep-clean:  ## clean everything untracked from the repository
	git clean -fdx

clean:  ## clean the repository, doesn't work on Windows
	rm -rf .coverage coverage cover htmlcov logs build dist *.egg-info .pytest_cache .mypy_cache identitychain/__pycache__ identitychain/tests/__pycache__


############################################################################################


.DEFAULT_GOAL := help
help:  ## doesn't work on Windows
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

print-%:
	@echo '$*=$($*)'

.PHONY: develop build install lint lints format fix check checks annotate type test tests coverage cov show-version patch minor major dist-build dist-check dist publish pub deep-clean clean help
