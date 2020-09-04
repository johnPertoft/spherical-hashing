# Spherical Hashing
A Tensorflow 2 implementation of [Spherical Hashing](https://ieeexplore.ieee.org/document/6248024).

Spherical hashing is a way to compute a binary encoding of a feature vector while still maintaining
spatial coherence. This binary encoding can then be used for an approximate nearest neighbor 
solution since the compactness of this representation allows for faster neighbor search.

## Usage
```
from spherical_hashing import train_spherical_hashing
import tensorflow as tf

x_train = tf.random.uniform(shape=(1000, 512), minval=-10.0, maxval=10.0)
sph_model = train_spherical_hashing_model(x_train, n_bits=32)

x_test = tf.random.uniform(shape=(100, 512), minval=-10.0, maxval=10.0)
bits = sph_model(x_test, apply_pack_bits=True)
```

## Installation
```
pip install tf-spherical-hashing
```

## Development

### Run tests
```
./scripts/run-tests.sh
```

### Publish pip package
```
./scripts/publish.sh
```
