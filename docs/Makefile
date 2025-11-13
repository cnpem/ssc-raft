# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-multiversion
SOURCEDIR     = .
BUILDDIR      = build

all: out html

out:
	mkdir -p build

doxygen:
	doxygen doxyconf

example:
	cd ../ && python3 setup.py install --user --cuda && cd docs/
	@for j in example/*.py; do python3 $$j; done


.PHONY: help Makefile all

html:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"
