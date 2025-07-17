# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Workaround for Sphinx epub3 builder on Python 3.13+ (imghdr removed)
if sys.version_info >= (3, 13):
    try:
        import sphinx.builders

        if hasattr(sphinx.builders, "epub3"):
            del sphinx.builders.epub3
    except Exception:
        pass

sys.path.insert(0, os.path.abspath("../../src"))


project = "recsys_lite"
copyright = "2025, Simbarashe Timire"
author = "Simbarashe Timire"
release = "0.1.4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",  # For Markdown support
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

html_theme_options = {
    "navigation_depth": 4,
    "show_nav_level": 2,
    "collapse_navigation": False,
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Disable epub output (Python 3.13+ imghdr removal workaround) -------------
epub_build = False
