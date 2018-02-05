import numpy as np

from ..data_processing import findObsInCell

def test_findObsInCell_2D():
    # Create a set of points on a grid
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    ids = np.arange(0, len(xx))
    
    # Randomly select some centers and radii
    centers = np.random.choice(len(xx), size=5)
    radii = np.random.rand(5) * 10
    
    for center, radius in zip(centers, radii):
        # Calculate distances and select those that should be inside
        distances = np.sqrt((xx - xx[center])**2 + (yy - yy[center])**2)
        inside = ids[np.where(distances <= radius)[0]]
        np.testing.assert_array_equal(inside,
                                      findObsInCell(ids,
                                                    np.array([xx, yy]).T,
                                                    np.array([xx[center], yy[center]]),
                                                    radius))