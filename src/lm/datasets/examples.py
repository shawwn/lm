import tensorflow as tf

"The only valid example formats accepted by the framework"


def read_example(example_proto) -> dict:
    features = {
        "id": tf.io.VarLenFeature(tf.int64),
        "content": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.VarLenFeature(tf.int64),
        "offset_start": tf.io.VarLenFeature(tf.int64),
        "offset_end": tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return {
        "id": tf.cast(parsed_features["id"], tf.uint64),
        "content": parsed_features["content"],
        # WARNING: remapping from target to targets
        "targets": tf.sparse.to_dense(tf.cast(parsed_features["target"], tf.int64)),
        "offset_start": tf.sparse.to_dense(
            tf.cast(parsed_features["offset_start"], tf.uint64)
        ),
        "offset_end": tf.sparse.to_dense(
            tf.cast(parsed_features["offset_end"], tf.uint64)
        ),
    }
