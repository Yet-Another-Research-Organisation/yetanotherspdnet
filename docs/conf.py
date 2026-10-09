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
sys.path.insert(0, os.path.abspath("_ext"))
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "dualpath",  # _ext/dualpath.py: tables pairing autograd / manual-backward ops
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

# -- API reference (autodoc) -------------------------------------------------
# The reference pages (docs/reference/*.md) are written by hand: each opens with
# the context and a summary table, then documents the objects with autodoc.
autoclass_content = "both"  # class docstring + __init__ parameters, once
autodoc_member_order = "bysource"
autodoc_typehints = "description"  # types next to each parameter, short signatures
autodoc_typehints_description_target = "documented_params"
autodoc_default_options = {"exclude-members": "__init__, __new__"}
add_module_names = False  # "BiMap", not "yetanotherspdnet.nn.base.BiMap"
python_display_short_literal_types = True
python_maximum_signature_line_length = 88  # long signatures: one parameter per line
autosummary_generate = False  # summary tables only, no stub pages

# Napoleon settings (NumPy style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True  # attributes as fields: no duplicate objects
napoleon_use_param = True
napoleon_use_rtype = False
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
html_theme = "furo"
html_title = "Yet Another SPDNet"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#3a4a8c",
        "color-brand-content": "#3a4a8c",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9fb0f0",
        "color-brand-content": "#9fb0f0",
    },
}
