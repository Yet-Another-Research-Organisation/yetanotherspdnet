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
    "dualpath",  # _ext/dualpath.py: dualpath-table, autograd-only-table, deprecated-table
    "diagrams",  # _ext/diagrams.py: light / dark pairs of the diagrams
]

# MyST Parser configuration (Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
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


def _skip_deprecated(app, what, name, obj, skip, options):  # noqa: ARG001
    """Leave deprecated aliases (marked ``__deprecated__``) out of autodoc."""
    if hasattr(obj, "__deprecated__"):
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_deprecated)


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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_diagrams"]

# -- Options for HTML output -------------------------------------------------
# pydata-sphinx-theme, dark by default, one accent colour per section
# (_static/custom.css keys the accent off <body data-section>, set by
# _templates/layout.html from the first path segment of the page).
html_theme = "pydata_sphinx_theme"
html_title = "yetanotherspdnet"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.png"
html_show_sourcelink = False
html_context = {
    "default_mode": "dark",
    "github_user": "Yet-Another-Research-Organisation",
    "github_repo": "yetanotherspdnet",
    "github_version": "main",
    "doc_path": "docs",
}
html_theme_options = {
    # logo-dark.png: the same logo on a light disc (the navy of the logo is too
    # dark for the dark page)
    "logo": {
        "text": "yetanotherspdnet",
        "alt_text": "yetanotherspdnet",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo-dark.png",
    },
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 2,
    "show_nav_level": 1,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "tango",
    "pygments_dark_style": "github-dark",
}
# no left sidebar on the landing page and the bibliography
html_sidebars = {"index": [], "references": []}
