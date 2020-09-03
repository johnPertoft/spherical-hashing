import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk


class SphericalHashingModel(tfk.Model):
    def __init__(self, pivots, radii):
        super().__init__()
        assert pivots.shape[0] == radii.shape[0], "Num pivots and radii must be equal."
        assert radii.shape.ndims == 1, "Expected 1d tensor for radii."
        self.pivots = tf.Variable(
            tf.convert_to_tensor(pivots),
            trainable=False,
            name="pivots")
        self.radii = tf.Variable(
            tf.convert_to_tensor(radii),
            trainable=False,
            name="radii")

    def call(self, inputs, apply_pack_bits=True):
        distances = compute_pairwise_distances(inputs, self.pivots)
        bits = distances - self.radii[tf.newaxis, :] <= 0
        bits = tf.cast(bits, tf.uint8, name="bits")
        if apply_pack_bits:
            bits = pack_bits(bits)
        return bits


def pack_bits(bits):
    """
    Pack tf.uint8 bit arrays into arrays of tf.int64 by interpreting blocks of 64 bits
    as an int64.
    Args:
        bits: A tf.uint8 `Tensor` of shape (None, n_bits). `n_bits` must be a multiple of 64.
    Returns:
        bits: A tf.int64 `Tensor` of shape (None, n_bits // 64)
    """
    n_bits = bits.shape[1]
    assert int(n_bits) % 64 == 0, "Number of bits must be a multiple of 64."
    powers_of_2 = tf.pow(tf.cast(2, tf.int64), tf.range(0, 64, dtype=tf.int64))
    bits = tf.reshape(bits, [-1, 64])
    bits = tf.reduce_sum(tf.cast(bits, tf.int64) * powers_of_2, axis=1)
    bits = tf.reshape(bits, [-1, n_bits // 64])
    return bits


def zero_diag(inputs):
    """
    Return inputs with a zeroed diagonal.
    Args:
        inputs: A float32 2d `Tensor`.
    Returns:
        inputs with a zeroed diagonal.
    """
    return inputs - tf.linalg.diag(tf.linalg.diag_part(inputs))


def compute_pairwise_distances(a, b):
    """
    Compute pairwise euclidean distances between row vectors.
    Args:
        a: A float32 `Tensor` of shape (n_a_rows, n_dims).
        b: A float32 `Tensor` of shape (n_b_rows, n_dims).
    Returns:
        dists: A float32 `Tensor` of shape (n_a_rows, n_b_rows).
    """
    a_sq_norm = tf.reduce_sum(a * a, axis=1, keepdims=True)
    b_sq_norm = tf.reduce_sum(b * b, axis=1, keepdims=True)
    dists_sq = a_sq_norm - 2 * (a @ tf.transpose(b)) + tf.transpose(b_sq_norm)
    dists = tf.sqrt(dists_sq)
    return dists


def compute_radii(distances, beta=0.05):
    """
    Compute the radius for every hypersphere given the pairwise distances
    to satisfy Eq. 6 in the paper. Also implements the heuristic described in
    section 3.5.
    Args:
        distances: A float32 `Tensor` of shape (n_pivots, n_inputs).
        beta: Float parameter controlling the degree of tolerance for breaking
              the balance partition criterion. Or in other words controlling
              the candidate set size.
    Returns:
        radii: A float32 `Tensor` of shape (n_pivots,).
    """
    n_pivots = tf.shape(distances)[0]
    n_inputs = tf.shape(distances)[1]

    # Compute the candidate set by considering a window around the median index.
    # See section 3.5 in the paper.
    sorted_distances = tf.sort(distances, direction="ASCENDING", axis=-1)
    candidate_set_half_size = tf.math.round(beta * tf.cast(n_inputs, tf.float32))
    candidate_set_half_size = tf.cast(candidate_set_half_size, tf.int32)
    median_index = n_inputs // 2
    s = median_index - candidate_set_half_size
    e = median_index + candidate_set_half_size + 1
    candidate_set = sorted_distances[:, s:e]
    
    # Compute the margins between the candidates.
    distance_differences = candidate_set[:, 1:] - candidate_set[:, :-1]
    max_diff_index = tf.argmax(distance_differences, axis=-1, output_type=tf.int32)
    candidate_index = tf.stack((tf.range(n_pivots), max_diff_index), axis=-1)

    # Set the radii as the average of the two candidates with the biggest margin
    # between each other.
    radii =  1 / 2 * (
        tf.gather_nd(candidate_set, candidate_index) + 
        tf.gather_nd(candidate_set, candidate_index + [0, 1]))
    
    return radii


def compute_radii_simple(distances):
    """
    Compute the radius for every hypersphere given the pairwise distances
    to satisfy Eq. 6 in the paper. Does not implement the heuristic described
    in section 3.5.
    """
    n_inputs = tf.shape(distances)[1]

    sorted_distances = tf.sort(distances, direction="ASCENDING", axis=-1)
    median_index = n_inputs // 2
    radii = sorted_distances[:, median_index]

    return radii


def compute_overlaps(distances, radii):
    """
    Compute the overlaps o_i_j, i.e. the number of datapoints that are contained in 
    both the i and j hyperspheres.
    Args:
        distance: A float32 `Tensor` of shape (n_pivots, n_inputs).
        radii: A float32 `Tensor` of shape (n_pivots,).
    Returns:
        overlaps: A float32 `Tensor` of shape (n_pivots, n_pivots) representing each o_i_j.
    """
    inside_sphere = tf.cast(distances <= radii[:, tf.newaxis], tf.float32)
    overlaps = tf.matmul(inside_sphere, tf.transpose(inside_sphere))
    return overlaps


def compute_forces(overlaps, pivots, n_inputs):
    """
    Compute the forces that should be applied to every pivot.
    Args:
        overlaps: A float32 `Tensor` of shape (n_pivots, n_pivots) representing each o_i_j.
        pivots: A float32 `Tensor` of shape (n_pivots, input_dim).
        n_inputs: Integer, the number of inputs.
    Returns:
        forces: A float32 `Tensor` of shape (n_pivots, input_dim).
    """
    n_pivots = tf.shape(pivots)[0]

    pivots_tiled = tf.tile(pivots[:, tf.newaxis, :], [1, n_pivots, 1])
    pivots_diff = pivots_tiled - pivots

    overlap_target = tf.cast(n_inputs / 4, tf.float32)
    forces_multiplier = 1 / 2 * (overlaps - overlap_target) / overlap_target
    forces = forces_multiplier[..., tf.newaxis] * pivots_diff
    forces = tf.reduce_mean(forces, axis=1)

    return forces


def compute_statistics(overlaps, n_inputs):
    """
    Compute mean and stddev of the overlaps, not counting the diagonal.
    Args:
        overlaps: A float32 `Tensor` of shape (n_pivots, n_pivots) representing each o_i_j.
        n_inputs: Integer, the number of inputs.
    Returns:
        abs_diff_mean: A float, the mean of absolute difference from the overlap target.
        stddev: A float, the stddev of the overlaps.
    """
    n_pivots = tf.shape(overlaps)[0]

    # We don't count the self overlaps.
    n_overlaps_to_count = n_pivots * (n_pivots - 1)
    n_overlaps_to_count = tf.cast(n_overlaps_to_count, tf.float32)

    overlap_target = tf.cast(n_inputs / 4, tf.float32)
    abs_diff = tf.abs(overlaps - overlap_target)
    abs_diff_mean = tf.reduce_sum(zero_diag(abs_diff)) / n_overlaps_to_count

    mean = tf.reduce_sum(zero_diag(overlaps)) / n_overlaps_to_count
    variance = tf.reduce_sum(zero_diag((overlaps - mean) ** 2)) / n_overlaps_to_count
    stddev = tf.sqrt(variance)

    return abs_diff_mean, stddev


@tf.function
def spherical_hashing_update_step(pivots, radii, inputs):
    """
    Computes new pivots and radii to better satisfy the constraints.
    Args:
        pivots: A float32 `Tensor` of shape (n_pivots, input_dim).
        radii: A float32 `Tensor` of shape (n_pivots,).
        inputs: A float32 `Tensor` of shape (n_inputs, input_dim).
    Returns:
        updated_pivots: A float32 `Tensor` of shape (n_pivots, input_dim).
        updated_radii: A float32 `Tensor` of shape (n_pivots,).
        overlaps_abs_diff_mean: A float, the mean of the absolute differences 
            of overlaps and overlaps target.
        overlaps_stddev: A float, the standard deviance of the overlaps.
    """
    n_inputs = tf.shape(inputs)[0]

    # Compute the initial state of overlaps of hyperspheres.
    distances = compute_pairwise_distances(pivots, inputs)
    overlaps = compute_overlaps(distances, radii)

    # Compute updated pivots and recompute distances.
    forces = compute_forces(overlaps, pivots, n_inputs)
    updated_pivots = pivots + forces
    updated_distances = compute_pairwise_distances(updated_pivots, inputs)
    
    # TODO: It seems like the compute_radii_simple works better? In terms of convergence at least.
    # Compute updated radii for updated pivots.
    updated_radii = compute_radii_simple(updated_distances)

    # Compute the statistics of the current overlaps.
    overlaps = compute_overlaps(distances, radii)
    overlaps_abs_diff_mean, overlaps_stddev = compute_statistics(overlaps, n_inputs)

    return updated_pivots, updated_radii, overlaps_abs_diff_mean, overlaps_stddev


def train_spherical_hashing_model(
    inputs,
    n_bits,
    epsilon_abs_diff_mean=0.1,
    epsilon_stddev=0.15,
    max_steps=100,
    seed=None):
    """
    The training procedure to produce a `SphericalHashingModel`.
    Args:
        inputs: A float32 `Tensor` of shape (n_inputs, input_dim). The training data.
        n_bits: Number of pivots (equivalently, number of hash bits).
        epsilon_abs_diff_mean: Error tolerance for overlaps diff mean.
        epsilon_stddev: Error tolerance for overlaps stddev.
        max_steps: The maximum number of update steps to take.
        seed: The random seed.
    """
    if seed is None:
        seed = 0

    n_inputs = tf.shape(inputs)[0]

    # Initialize pivots.
    pivots = []
    n_samples = 5
    for i in range(n_bits):
        indices = tf.range(n_inputs)
        indices = tf.random.shuffle(indices, seed + i)
        indices = indices[:n_samples]
        samples = tf.gather(inputs, indices)
        sample_mean = tf.reduce_mean(samples, axis=0, keepdims=True)
        pivots.append(sample_mean)
    pivots = tf.concat(pivots, axis=0)

    # Initialize radii.
    distances = compute_pairwise_distances(pivots, inputs)
    radii = compute_radii_simple(distances)

    for _ in range(max_steps):
        pivots, radii, overlaps_abs_diff_mean, overlaps_stddev = \
            spherical_hashing_update_step(pivots, radii, inputs)

        overlap_target = tf.cast(n_inputs / 4, tf.float32)
        diff_condition = overlaps_abs_diff_mean <= epsilon_abs_diff_mean * overlap_target
        stddev_condition = overlaps_stddev <= epsilon_stddev * overlap_target
        if diff_condition and stddev_condition:
            break

    return SphericalHashingModel(pivots, radii)
