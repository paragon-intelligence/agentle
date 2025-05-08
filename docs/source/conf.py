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

language = "en"  # Explicitly set the language

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

# Copy the logo to the build directory
html_logo = "../logo.png"
html_favicon = "../logo.png"

# Ensure the right output paths for GitHub Pages
html_baseurl = "https://paragon-intelligence.github.io/agentle/"

# Disable use of file modification times in output
html_last_updated_fmt = ""

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Search settings - disabling snowballstemmer to fix the issue
html_search_language = "en"
html_search_options = {"type": "default"}

# Disable stemmer for search
html_search_scorer = ""
html_search_options = {"dict": "english"}

# We'll keep the search functionality enabled
html_use_index = True
html_use_searchindex = True
html_search_enabled = True
