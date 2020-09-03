/* 
Infeed library. Supported infeeds:
- ExampleGenerator
- TFRecordDataset: feeds the training process using a set of files written in the tfrecord format
*/
{
    ExampleGenerator(producer, seed=1337, batch_size=32) :: {
        // An infeed that Generates from a tf.Example producer
        kind: 'lm.infeeds.ExampleGenerator',
        producer: producer,
        batch_size: batch_size,
        max_sequence_length: producer.max_sequence_length
    },

    TFRecordDataset()::{

    },
}