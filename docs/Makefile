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

.PHONY: help Makefile all

html:
	@$(SPHINXMBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

branch:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"
