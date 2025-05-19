# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "Agentle"
copyright = "2025, Agentle Team"
author = "Agentle Team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

# Configure autosummary to automatically generate stub files
autosummary_generate = True
autosummary_imported_members = True

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"  # Changed from sphinx_rtd_theme to furo
html_static_path = ["_static"]
html_logo = "../../docs/logo.png"
html_favicon = "../../docs/favicon.ico"

# Configure furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    # "announcement": "This is a preview of Agentle's documentation (Release date: May 16, 2025)",
    "light_css_variables": {
        "color-brand-primary": "#2980B9",
        "color-brand-content": "#2980B9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4DA3E2",
        "color-brand-content": "#4DA3E2",
    },
}

# -- Extension configuration -------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
add_module_names = False

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx mapping ----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}
