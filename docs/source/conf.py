# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pulse Optimization'
copyright = '2023, Roman Wixinger'
author = 'Roman Wixinger'

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('../..')))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.linkcode",
]

autosummary_generate = True

napoleon_google_docstring = True
napoleon_include_private_with_doc = True

autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'qiskit': ('https://qiskit.org/', None),
    'typing': ('https://docs.python.org/3/library/typing.html', None),
    'pandas': ('https://pandas.pydata.org/', None)
}


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/romanwixinger/pulse-optimization/tree/documentation/%s.py" % filename
