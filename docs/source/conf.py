# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Add the project root directory to the path so Sphinx can find the package
sys.path.insert(0, os.path.abspath("../.."))

project = "Agentle"
copyright = "2025, Arthur Brenno"
author = "Arthur Brenno"
release = "0.0.13"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
pygments_style = "default"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc configuration --------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autoclass_content = "both"
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "sidebar_hide_name": False,
}
html_static_path = ["_static"]
html_title = f"{project} {release} Documentation"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
