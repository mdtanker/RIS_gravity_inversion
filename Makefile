# install and clean
PROJECT=RIS_gravity_inversion
STYLE_CHECK_FILES= . #$(PROJECT)
#
#
#
# INSTALL
#
#
#
install:
	pip install -e .
	# pip install --upgrade git+https://github.com/fatiando/verde

conda_install:
	mamba env create --file env/environment.yml --name $(PROJECT) --force

conda_update:
	mamba env update --file env/environment.yml --name $(PROJECT) --prune
#
#
#
# STYLE
#
#
#
format: isort black

check: isort-check black-check flake8

black:
	black --line-length 88 $(STYLE_CHECK_FILES)

black-check:
	black --line-length 88 --check $(STYLE_CHECK_FILES)

isort:
	isort $(STYLE_CHECK_FILES)

isort-check:
	isort --check $(STYLE_CHECK_FILES)

flake8:
	flake8p --max-line-length 88 $(STYLE_CHECK_FILES) --exclude=.ipynb_checkpoints

#
#
#
# TESTING
#
#
#
test:
	pytest --cov=. --no-cov