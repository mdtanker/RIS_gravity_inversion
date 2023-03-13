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
install: delete_env
	mamba env create --file environment.yml --name $(PROJECT)

delete_env:
	mamba remove --name $(PROJECT) --all --yes

conda_update:
	mamba env update --file environment.yml --name $(PROJECT)
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