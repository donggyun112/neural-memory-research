import numpy as np

from fetch_retinotopy import normalize_unit, pca_2d


def test_pca_2d_recovers_a_flat_sheet_exactly():
    # points lie exactly in the z=0 plane -- PCA should recover x,y (up to
    # rotation/sign) and discard z entirely.
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, size=(50, 2))
    points = np.column_stack([xy, np.zeros(50)])
    projected = pca_2d(points)
    assert projected.shape == (50, 2)
    # pairwise distances in the projection should match the original xy plane
    from scipy.spatial.distance import pdist
    assert np.allclose(pdist(projected), pdist(xy), atol=1e-8)


def test_normalize_unit_maps_to_zero_one_range():
    coords = np.array([[0.0, 5.0], [10.0, 15.0], [5.0, 10.0]])
    normalized = normalize_unit(coords)
    assert normalized.min() == 0.0
    assert np.isclose(normalized.max(), 1.0)
