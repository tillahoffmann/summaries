master_doc = 'README'
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
]
project = 'summaries'
napoleon_custom_sections = [('Returns', 'params_style')]
plot_formats = [
    ('png', 144),
]
