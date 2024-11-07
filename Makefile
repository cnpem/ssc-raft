RUNDIR=	sscRaft sscRaft/ cuda/

all: install

install:
	python3 -m pip install -v .

pip-dev:
	pip install . --no-deps --no-build-isolation

dev:
	python3 setup.py install

dgx:
	python3 setup.py install --user

clean:
	rm -fr _skbuild/ build/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*~; done

