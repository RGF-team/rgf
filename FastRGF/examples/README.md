# Examples

You can learn how to use FastRGF by these examples.

Note that for these small examples, the running time with multithreading may be slower than with single-threading due to the overhead it introduces.
However, for large datasets, one can observe an almost linear speedup.

FastRGF can directly handle high-dimensional sparse features in the libsvm format as in [binary_classification example](./binary_classification).
This is the recommended format to use when the dataset is relatively large (although some other formats are supported).
