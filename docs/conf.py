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
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_title = "Yet Another SPDNet"

# Sphinx Awesome Theme options
html_theme_options = {
    "show_breadcrumbs": True,
    "show_prev_next": True,
    "show_scrolltop": True,
    "extra_header_link_icons": {
        "GitHub": {
            "link": "https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet",
            "icon": (
                '<svg height="26px" style="margin-top:-2px;display:inline" '
                'viewBox="0 0 45 44" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill-rule="evenodd" clip-rule="evenodd" '
                'd="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 17.736 '
                '14.853 20.608 1.087.2 1.483-.47 1.483-1.047 0-.516-.019-1.881-.03-3.697-6.04 '
                '1.31-7.315-2.91-7.315-2.91-.988-2.51-2.412-3.178-2.412-3.178-1.972-1.346.149-1.32.149-1.32 '
                '2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 1.803.197-1.403.739-2.36 '
                '1.349-2.903-4.735-.542-9.713-2.367-9.713-10.542 0-2.33.828-4.233 2.19-5.73-.221-.544-.961-2.74.206-5.713 '
                '0 0 1.795-.573 5.885 2.193 1.705-.48 3.547-.72 5.37-.72 1.823 0 3.665.24 5.37.72 4.09-2.766 '
                '5.885-2.193 5.885-2.193.367 2.973-.027 5.169.206 5.713 1.362 1.497 2.19 3.4 2.19 5.73 0 '
                '8.216-4.995 10.001-9.76 10.542.76.694 1.44 2.063 1.44 4.158 0 3.007-.057 5.437-.075 6.178.909.52 '
                '1.52 1.21 1.52 2.58 0 1.86-.27 3.32-.81 3.86C41.7 40.683 48 32.54 48 22.647 48 10.65 38.28.927 '
                '26.29.927z" fill="currentColor"/>'
                '</svg>'
            ),
        }
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
