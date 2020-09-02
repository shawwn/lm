import tensorflow as tf
import farmhash

"The only valid example formats accepted by the framework"

PreProcessedTextLine = collections.namedtuple(
    "PreProcessedTextLine", ["id", "content", "target", "offset_start", "offset_end"]
)


def _uint64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=np.int64(np.array(value, dtype=np.uint64)))
    )


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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

def read_example(example_proto, max_seq_len=1024) -> dict:
    features = {
        "id": tf.VarLenFeature(tf.uint64, default=-1),
        "content": tf.VarLenFeature(tf.bytes, default=0),
        "target": tf.VarLenFeature(tf.uint64, default=0),
        "offset_start": tf.VarLenFeature(tf.uint64, default=0),
        "offset_end": tf.VarLenFeature(tf.uint64, default=0),
    }
    return tf.parse_single_example(example_proto, features)


def create_example(features: PreProcessedTextLine) -> tf.train.Example:
    feature = {
        "id": _uint64_feature([features.id]),
        "content": _bytes_feature(features.content.encode("utf-8")),
        "target": _uint64_feature(features.target),
        "offset_start": _uint64_feature(features.offset_start),
        "offset_end": _uint64_feature(features.offset_end),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def transform_many_and_write_one_tfrecord(job):
    tokenizer, sources, dst = job
    with tf.io.TFRecordWriter(dst) as w:
        for source in sources:
            for features in batch_tokenizer(tokenizer, source):
                example = create_example(PreProcessedTextLine(*features))
                w.write(example.SerializeToString())
    return len(sources)

def batch_tokenizer(tokenizer, txtfile_location):
    # just convert to the token ids, we will do adaptative padding on training time.
    lines = [
        l.decode("utf-8") for l in tf.io.gfile.GFile(txtfile_location, "rb").readlines()
    ]
    uids = [farmhash.fingerprint64(line) for line in lines]
    batches = tokenizer.batch_encode_plus(
        lines,
        return_token_type_ids=True,
        pad_to_max_length=False,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
        verbose=False,
    )

    return zip(
        uids,
        lines,
        batches["input_ids"],
        [[start for start, end in offsets] for offsets in batches["offset_mapping"]],
        [[end for start, end in offsets] for offsets in batches["offset_mapping"]],
    )
