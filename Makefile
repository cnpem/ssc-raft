RUNDIR=	sscRaft sscRaft/ cuda/ 

all: install

install:
	pip install . 

dev:
	python3 setup.py install 

dgx:
	python3 setup.py install --user

clean:
	rm -fr _skbuild/ build/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*~; done

