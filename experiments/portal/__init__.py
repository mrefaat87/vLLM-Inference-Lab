"""Static portal builder.

Produces ``command_center.html`` (manifest list) and ``results_explorer.html``
(empirical vs roofline overlay). No bundler, no npm — Python writes HTML
that includes a small inline JS for interactivity and Chart.js (via CDN
with SRI) for plotting.
"""
