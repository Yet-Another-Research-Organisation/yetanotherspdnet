"""Sphinx configuration for Yet Another SPDNet."""

import os
import sys
from datetime import datetime


# Add source to path
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Yet Another SPDNet"
copyright = f"{datetime.now().year}, Ammar Mian, Florent Bouchard, Guillaume Ginolhac, Matthieu Gallet"
author = "Ammar Mian, Florent Bouchard, Guillaume Ginolhac, Matthieu Gallet"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "autoapi.extension",
]

# MyST Parser configuration (Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# AutoAPI configuration (automatic API documentation)
autoapi_type = "python"
autoapi_dirs = ["../src/yetanotherspdnet"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_keep_files = True
autoapi_add_toctree_entry = False

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Intersphinx mapping (link to other projects' documentation)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
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
html_theme = "alabaster"
html_static_path = ["_static"]
html_title = "Yet Another SPDNet"

# Alabaster theme options
html_theme_options = {
    "description": "A robust implementation of SPDNet for SPD matrices",
    "github_user": "Yet-Another-Research-Organisation",
    "github_repo": "yetanotherspdnet",
    "github_banner": True,
    "github_button": True,
    "github_type": "star",
    "show_related": False,
    "note_bg": "#FFF59C",
    "show_powered_by": False,
    "sidebar_collapse": True,
    "extra_nav_links": {
        "GitHub": "https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet",
        "Issue Tracker": "https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet/issues",
    },
}

# Logo and favicon (optional - uncomment if you add these)
# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

# Additional CSS (optional)
# html_css_files = ["custom.css"]

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
}

latex_documents = [
    (
        "index",
        "YetAnotherSPDNet.tex",
        "Yet Another SPDNet Documentation",
        author,
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (
        "index",
        "yetanotherspdnet",
        "Yet Another SPDNet Documentation",
        [author],
        1,
    )
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        "index",
        "YetAnotherSPDNet",
        "Yet Another SPDNet Documentation",
        author,
        "YetAnotherSPDNet",
        "A robust and tested implementation of SPDNet learning models.",
        "Miscellaneous",
    ),
]
