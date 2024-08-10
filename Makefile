PROJECT=RIS_gravity_inversion
STYLE_CHECK_FILES=.

####
####
# install commands
####
####

install:
	pip install -e .

remove:
	mamba remove --name $(PROJECT) --all

conda_install:
	mamba env create --file environment.yml --name $(PROJECT)

conda_update:
	mamba env update --file environment.yml --name $(PROJECT) --prune

####
####
# style commands
####
####

format:
	ruff format $(STYLE_CHECK_FILES)

check:
	ruff check --fix $(STYLE_CHECK_FILES)

lint:
	pre-commit run --all-files

pylint:
	pylint $(PROJECT)

style: format check lint

####
####
# chore commands
####
####

clean:
	find . -name 'tmp_*.pickle' -delete

# find . -name '*.log' -delete
# find . -name '*.lock' -delete
# find . -name '*.pkl' -delete
# find . -name '*.sqlite3' -delete
# find . -name '*.coverage' -delete
