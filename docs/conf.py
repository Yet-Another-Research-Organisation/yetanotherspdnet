"""Sphinx configuration for Yet Another SPDNet."""

import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version


# Add source to path
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Yet Another SPDNet"
author = "Ammar Mian, Florent Bouchard, Guillaume Ginolhac, Matthieu Gallet"
copyright = f"{datetime.now().year}, {author}"
try:
    release = package_version("yetanotherspdnet")
except PackageNotFoundError:  # docs built without installing the package
    release = "dev"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "autoapi.extension",
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST Parser configuration (Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "linkify",
    "smartquotes",
    "substitution",
]
myst_heading_anchors = 3

# AutoAPI configuration (API reference generated from docstrings)
autoapi_type = "python"
autoapi_dirs = ["../src/yetanotherspdnet"]
autoapi_root = "reference"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
# Most classes document their constructor arguments in __init__: show both the
# class docstring (description + attributes) and the __init__ one (parameters).
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_own_page_level = "module"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
suppress_warnings = ["autoapi.python_import_resolution"]

# Napoleon settings (NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Intersphinx mapping (link to other projects' documentation)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Source files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "Yet Another SPDNet"
html_static_path = []
html_theme_options = {
    "github_url": "https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {"default_mode": "auto"}
