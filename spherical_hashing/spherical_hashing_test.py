import numpy as np
import tensorflow as tf

from .spherical_hashing import compute_forces
from .spherical_hashing import compute_statistics
from .spherical_hashing import compute_overlaps
from .spherical_hashing import compute_pairwise_distances
from .spherical_hashing import compute_radii
from .spherical_hashing import compute_radii_simple
from .spherical_hashing import pack_bits
from .spherical_hashing import train_spherical_hashing_model
from .spherical_hashing import zero_diag


def test_pack_bits():
    bits = np.zeros((2, 64 * 4), dtype=np.uint8)
    
    bits[0, 2] = 1  # 4
    bits[0, 4] = 1  # 16
    bits[0, 8] = 1  # 256
    
    bits[1, 32] = 1  # 4294967296
    bits[1, 64] = 1  # 1
    bits[1, 128] = 1  # 1
    bits[1, 192] = 1  # 1

    bits = tf.convert_to_tensor(bits)
    bits = pack_bits(bits)

    expected_bits = np.array([
        [276, 0, 0, 0],
        [4294967296, 1, 1, 1]
    ], dtype=np.int64)

    np.testing.assert_array_equal(bits, expected_bits)


def test_zero_diag():
    x = tf.convert_to_tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    expected_zeroed_diag = tf.convert_to_tensor([
        [0, 2, 3, 4],
        [5, 0, 7, 8],
        [9, 10, 0, 12],
        [13, 14, 15, 0]
    ])

    zeroed_diag = zero_diag(x)

    np.testing.assert_array_equal(zeroed_diag, expected_zeroed_diag)


def test_compute_pairwise_distances():
    x = tf.convert_to_tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7.1, 2]
    ])

    expected_distances = tf.convert_to_tensor([
        [0.00000, 2.82843, 5.65685, 6.10000],
        [2.82843, 0.00000, 2.82843, 4.56180],
        [5.65685, 2.82843, 0.00000, 4.51774],
        [6.10000, 4.56180, 4.51774, 0.00000]])

    distances = compute_pairwise_distances(x, x)

    np.testing.assert_allclose(distances, expected_distances, rtol=1e-5)


def test_compute_radii():
    # Already sorted distance matrix to more easily see which should end up in the
    # candidate set.
    distances = tf.convert_to_tensor([
        [ 6.12,  8.79, 10.61, 10.85, 11.48, 12.09, 12.54, 13.04, 13.25, 15.19, 16.54],
        [ 3.76,  4.95,  5.02,  9.84,  9.98, 10.27, 10.61, 10.75, 11.66, 14.05, 14.4 ],
        [ 2.47,  3.88,  4.34,  5.86,  5.88,  7.57,  8.03,  9.57, 11.48, 12.23, 14.6 ],
        [ 2.69,  4.95,  5.32,  5.88,  7.11,  8.32,  8.36,  9.79, 10.  , 10.75, 12.09]])

    # Choose beta such that we get a candidate set size of three (one on each side 
    # of the median).
    n_inputs = distances.shape[1]
    beta = 1 / n_inputs

    radii = compute_radii(distances, beta=beta)

    # Given the candidate set size of three (that should happen) we expect the 
    # following distances from the distance matrix to be averaged into the 
    # final radii because they have the largest margin between them out of the
    # candidates in the candidate set. See Eq. 10, 11, 12 in paper.
    expected_radii = np.array([
        0.5 * (11.48 + 12.09),
        0.5 * (10.27 + 10.61),
        0.5 * (5.88 + 7.57),
        0.5 * (7.11 + 8.32)
    ])

    np.testing.assert_allclose(radii, expected_radii, rtol=1e-5)


def test_compute_radii_simple():
    # Already sorted distance matrix to more easily see which should be selected as
    # the radii.
    distances = tf.convert_to_tensor([
        [ 6.12,  8.79, 10.61, 10.85, 11.48, 12.09, 12.54, 13.04, 13.25, 15.19, 16.54],
        [ 3.76,  4.95,  5.02,  9.84,  9.98, 10.27, 10.61, 10.75, 11.66, 14.05, 14.4 ],
        [ 2.47,  3.88,  4.34,  5.86,  5.88,  7.57,  8.03,  9.57, 11.48, 12.23, 14.6 ],
        [ 2.69,  4.95,  5.32,  5.88,  7.11,  8.32,  8.36,  9.79, 10.  , 10.75, 12.09]])

    radii = compute_radii_simple(distances)

    expected_radii = np.array([
        12.09,
        10.27,
        7.57,
        8.32
    ])

    np.testing.assert_allclose(radii, expected_radii, rtol=1e-5)


def test_compute_overlaps():
    # Sorted distances to make it easier to see which are inside the hyperspheres
    # defined by these distances and the radii below.
    distances = tf.convert_to_tensor([
        [ 6.12,  8.79, 10.61, 10.85, 11.48, 12.09, 12.54, 13.04, 13.25, 15.19, 16.54],
        [ 3.76,  4.95,  5.02,  9.84,  9.98, 10.27, 10.61, 10.75, 11.66, 14.05, 14.4 ],
        [ 2.47,  3.88,  4.34,  5.86,  5.88,  7.57,  8.03,  9.57, 11.48, 12.23, 14.6 ],
        [ 2.69,  4.95,  5.32,  5.88,  7.11,  8.32,  8.36,  9.79, 10.  , 10.75, 12.09]])

    radii = tf.convert_to_tensor([
        12.0,
        10.0,
        9.0,
        7.0
    ])

    overlaps = compute_overlaps(distances, radii)

    expected_overlaps = np.array([
        [5., 5., 5., 4.],
        [5., 5., 5., 4.],
        [5., 5., 7., 4.],
        [4., 4., 4., 4.]
    ])

    np.testing.assert_array_equal(overlaps, expected_overlaps)


def test_compute_forces():
    pivots = tf.convert_to_tensor([
        [1.0, 3.0],
        [3.0, 1.0],
        [5.0, 3.0]
    ])

    overlaps = tf.convert_to_tensor([
        [5., 2., 3.],
        [2., 5., 3.],
        [3., 3., 6.]
    ])

    forces = compute_forces(overlaps, pivots, n_inputs=8)

    expected_forces = np.array([
        np.mean([
            [0., 0.],
            [0., 0.],
            [-1, 0.]
        ], axis=0),

        np.mean([
            [0., 0.],
            [0., 0.],
            [-0.5, -0.5]
        ], axis=0),

        np.mean([
            [1., 0.],
            [0.5, 0.5],
            [0., 0.]
        ], axis=0)
    ])

    np.testing.assert_allclose(forces, expected_forces, rtol=1e-5)


def test_compute_statistics():
    n_inputs = 8
    overlaps = tf.convert_to_tensor([
        [5., 2., 3.],
        [2., 5., 3.],
        [3., 3., 6.]
    ])

    overlaps_abs_diff_mean, overlaps_stddev = compute_statistics(overlaps, n_inputs)

    non_diagonal_overlaps = np.array([2, 3, 2, 3, 3, 3])

    expected_overlaps_abs_diff_mean = (non_diagonal_overlaps - n_inputs / 4).mean()
    np.testing.assert_almost_equal(overlaps_abs_diff_mean, expected_overlaps_abs_diff_mean)

    mean = np.mean(non_diagonal_overlaps)
    expected_stddev = np.sqrt(np.mean((non_diagonal_overlaps - mean) ** 2))
    np.testing.assert_almost_equal(overlaps_stddev.numpy(), expected_stddev)


def test_spherical_hashing_update_step():
    # TODO: What can we assert? That one step improves things?
    pass


def test_train_spherical_hashing_model():
    n_dims = 1024
    n_bits = 64
    n_train = 25000

    def get_inputs(n):
        return np.random.uniform(low=-10.0, high=10.0, size=(n, n_dims))

    x_train = get_inputs(n_train)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

    model = train_spherical_hashing_model(
        x_train,
        n_bits=n_bits,
        epsilon_abs_diff_mean=0.1,
        epsilon_stddev=0.15,
        max_steps=50,
        seed=1)

    # Assert that points from the training set are roughly in 50% of the hyperspheres.
    n = 100
    bits = model(x_train[:n], apply_pack_bits=False)
    bits = tf.cast(bits, tf.float32)
    mean = tf.reduce_mean(bits).numpy()
    expected_mean = 0.5
    assert np.abs(mean - expected_mean) < 0.05

    # TODO: Should make sure that points close to each other get very similar binary representations.
    # TODO: Should make sure that points that are not very close to each other get very different binary representations.
    # TODO: Add some assertions on the number of overlaps?
