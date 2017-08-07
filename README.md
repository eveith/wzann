# wzAnn

A library and command-line tools for artificial neural networks, providing
perceptrons and simple recurring neural networks.

## State

This library is currently usable, but missing interesting features except for
the usage of REvol, the multi-part evolutionary training algorithm.

There is currently no API stability promise. Expect that for version 1.0.
Until this milestone is reached, i.e., for every 0.MINOR release, expect a
breaking change. Patchlevel increments (i.e., 0.MINOR.PATCH) do not break the
API, and only fix bugs or add functionality in a backwards-compatible manner.

## Roadmap

### 0.7

  * Add manpages for CLI
  * Reorganize namespace:
      * `wzann`, the toplevel namespace, will contain the major patterns that
        emit an ANN. The "Pattern" suffix will be removed. E.g.,
        the current `wzann::ElmanNetworkPattern` will be converted to
        wzann::ElmanNetwork. Additionally, each NeuralNetworkPattern will gain
        a `createNeuralNetwork() const` method that simply emits a new ANN.
      * `wzann::model` will contain the structure classes, such as the actual
        `NeuralNetwork` class and its content classes, `Layer` and `Neuron`.
        The `ActivationFunction` enum and functions will also be housed here.
      * `wzann::serialization` will contain the helper functions that are
        concerned with serialization, such as to_variant/from_variant,
        to_json/from_json, and the `ClassRegistry`.
      * `wzann::train` will house all training algorithms and related
        functionality, as well as the `TrainingSet` and `TrainingItem` classes

### 0.8

  * Add the Nadam gradient descent algorithm
  * Refactors training algorithms for one common evaluation loop that can
    be parallelized
  * Add multi-threading functionality

### 0.9

  * Add Back-Propagation Through Time
  * Add LSTM

### 1.0

  * Freeze C++ API
  * Freeze CLI invocation API

## Installation and Requirements

wzAnn requires the following tools and libraries to be present:

  * CMake
  * A modern C++ compiler that understands C++14
  * wzalgorithm >= 0.8
  * LibVariant >= 1.0.0
  * Boost >= 1.54.0
  * (Optional) GTest, the Google Unit Testing framework
  * (Optional) Bats, to test the CLI utilities

Building wzAnn is very straightforwards:

    mkdir Build && cd Build
    cmake ..
    make && make test && make install

You should check for additional cmake command flags, though.

## License

wzAnn is licensed under GNU GPLv3. See the file `COPYING` for details.
`src/enum.h` is part of Better Enums, released under the BSD 2-clause license.
See <http://github.com/aantron/better-enums> for details.
