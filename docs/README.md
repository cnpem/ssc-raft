# Documentation

This repository's documentation is built using Doxygem, Sphinx and Breathe extension. Both Doxygen and Sphinx are used as documention generators, i.e., they read [docstrings](https://en.wikipedia.org/wiki/Docstring) to generate the documentation. The docstring must follow a pattern for Doxygen and another pattern for Sphinx. Breathe extension is used as bridge between Doxygen and Sphinx.

## Intallation

First of all, it is necessary to install Doxygen, Sphinx, Read the Docs theme and Breathe. The commands are:

```bash
sudo apt install doxygen
apt-get install python3-sphinx
pip install sphinx-rtd-theme
pip install breathe
```

## Usage

Once the docstrings are written, the documentation is generated using the following commands inside `docs/`:
```bash
doxygen doxyconf
make html
```
The folder `build/html/`, containing the documentation, will be created. The documentation can be seen using Firefox with the command
```bash
firefox build/html/index.html
```

**Note**: The folder `build/` is untracked by Git because it is too large.

## Configuration

Doxygen is used to autogenerate, in xml format, the documentation for CUDA files in `source/xml/`. Since Doxygen has no support for CUDA, it will be threated as C++. To do so, the Doxygen configuration file `doxyconf` must have some parameters changed as follows:
```
GENERATE_XML        = YES
XML_OUTPUT          = source/xml     # create xml/ folder inside source/
EXTENSION_MAPPING   = cu=c++
FILE_PATTERNS       = *.cu \
                      *.c \
                      *.cpp \
                      ...
```

The `INPUT` variable in the configuration file `doxyconf` must be changed to add source files folders
```
INPUT               = ../cuda/src/ ../cuda/inc/ ../cuda/inc/common/ ../sscPrain/ ../sscPrain/prain #your source files paths
```

The `source/conf.py` is the Sphinx configuration file. It has the autodoc, napoleon, Read the Docs theme and Breathe extensions. [Autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) is used to write phyton docstring in the documentation. [Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) enables autodoc to read docstrings written in the [Numpy](https://numpydoc.readthedocs.io/en/latest/format.html) or [Google](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) style guide docstrings. Read the Docs theme extension must be added in order to use Read the Docs theme. [Brethe](https://breathe.readthedocs.io/en/latest/) reads the files inside `source/xml/` enabling Sphinx to autogenerate documentation from CUDA docstrings, since autodoc can autogenerate documentation only for python. Then, the `extensions` variable in `conf.py` is set as:
```python
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme', 'breathe']
```

The `folder_path` variable in the configuration file `source/conf.py` must be changed to add source files folders
```python
folder_path = '../../sscPrain/' #your source files paths
```

## Tips

We suggest one to use the "Doxygen Documentation Generator" and "Python Docstring Generator" VSCode extensions to type the docstring.

Some useful spetial commands for Doxygen are `\a` that is used to display the next word in italics. Then it is used to cite parameters. `\f$` has the same meaning as `$` in latex, i.e., it is used to start and finish equations. `\note` is used to write a note. There are [more](https://www.doxygen.nl/manual/commands.html) special commands for Doxygen.