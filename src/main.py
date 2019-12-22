import tensorflow as tf
from lbow import LatentBow
from train import Controller
from config import config
from dataloader import Dataset


def main():
    tf.compat.v1.disable_eager_execution()

    # dataset
    dset = Dataset(config)
    dset.build()
    config.vocab_size = len(dset.word2id)
    config.dec_start_id = dset.word2id["_GOO"]
    config.dec_end_id = dset.word2id["_EOS"]
    config.pad_id = dset.pad_id
    config.stop_words = dset.stop_words

    model = LatentBow(config)
    with tf.compat.v1.variable_scope(config.model_name):
        model.build()

    # controller
    controller = Controller(config)
    controller.train(model, dset)


if __name__ == '__main__':
    main()
