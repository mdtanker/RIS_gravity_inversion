# Build, package, test, and clean
PROJECT=RIS_gravity_inversion
STYLE_CHECK_FILES= . #$(PROJECT)

install:
	mamba env create --file env/environment.yml

delete_env:
	mamba remove --name RIS_gravity_inversion --all --yes

conda_env: delete_env
	mamba create --name RIS_gravity_inversion --yes \
	python=3.9 \
	pygmt \
	geopandas \
	geoviews \
	numpy \
	pandas \
	snakeviz \
	scipy \
	matplotlib \
	pyproj \
	pyvista \
	harmonica \
	verde \
	xarray \
	tqdm \
	rioxarray \
	openssl \
	seaborn \
	ipyvtklink \
	ipykernel \

conda_yml:
	rm env/environment.yml -f
	mamba env export --name RIS_gravity_inversion --from-history --no-build > env/environment.yml

conda_update:
	mamba env update --file env/environment.yml

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
	flake8p --max-line-length 88 $(STYLE_CHECK_FILES)
