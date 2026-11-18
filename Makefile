RUNDIR=	sscRaft sscRaft/ cuda/

all: install

install:
	python3 -m pip install -v .

pip-dev:
	pip install . --no-deps --no-build-isolation

clean:
	rm -fr _skbuild/ build/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*~; done

