RUNDIR=	sscRaft sscRaft/ cuda/ example/

all: install

install:
	python3 setup.py install --user

clean:
	rm -fr build/ *.egg-info/ dist/	*~
	@for j in ${RUNDIR}; do rm -rf $$j/*.pyc; rm -rf $$j/*~; done

