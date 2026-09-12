"""Shared signed-value palette for financial heatmaps."""


def signed_heatmap_cmap():
    """Muted green for negative values, white at zero, muted red for positive.

    Use symmetric limits around zero; missing values remain visibly grey.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "financial_signed", ["#6A9589", "#FFFFFF", "#BC7272"], N=257,
    )
    return cmap.with_extremes(bad="#E4E7EA")
