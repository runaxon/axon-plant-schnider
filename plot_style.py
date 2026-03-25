"""
Shared matplotlib dark theme for all demo/evaluation plots.

Import and call apply() before any plt calls:

    from plot_style import apply, COLORS
    apply()
"""

import matplotlib.pyplot as plt

COLORS = {
    'bg':      '#141414',
    'surface': '#1e1e1e',
    'border':  '#2a2a2a',
    'text':    '#d4d4d4',
    'muted':   '#6a6a6a',
    'teal':    '#57c4b8',
    'orange':  '#d4956a',
    'green':   '#8ec07c',
    'red':     '#cc6666',
    'purple':  '#b294bb',
}


def apply() -> None:
    """Apply the dark monospace theme to matplotlib globally."""
    c = COLORS
    plt.rcParams.update({
        'figure.facecolor': c['bg'],
        'axes.facecolor':   c['surface'],
        'axes.edgecolor':   c['border'],
        'axes.labelcolor':  c['muted'],
        'xtick.color':      c['muted'],
        'ytick.color':      c['muted'],
        'text.color':       c['text'],
        'legend.facecolor': c['surface'],
        'legend.edgecolor': c['border'],
        'grid.color':       c['border'],
        'grid.linestyle':   '-',
        'axes.grid':        True,
        'font.family':      'monospace',
    })
