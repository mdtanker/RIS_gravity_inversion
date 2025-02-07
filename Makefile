PROJECT=RIS_gravity_inversion

####
####
# install commands
####
####

install:
	pip install -e .

remove:
	mamba remove --name $(PROJECT) --all

create:
	mamba env create --file environment.yml --name $(PROJECT)

update:
	mamba env update --file environment.yml --name $(PROJECT) --prune

####
####
# style commands
####
####

check:
	pre-commit run --all-files

pylint:
	pylint $(PROJECT)

style: check pylint

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
