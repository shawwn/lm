/* 
Infeed library. Supported infeeds:
- ExampleGenerator
- TFRecordDataset: feeds the training process using a set of files written in the tfrecord format
*/
{

    ExampleGenerator(producer, seed=1337) :: {
        // An infeed that Generates from a tf.Example producer
        kind: 'lm.infeeds.ExampleGenerator',
        producer: producer
    },

    TFRecordDataset()::{

    },
}