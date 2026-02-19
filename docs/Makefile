# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXMBUILD  ?= sphinx-multiversion
SPHINXBUILD   ?= sphinx-build
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
	@$(SPHINXMBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

branch:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"
