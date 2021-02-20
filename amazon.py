import os
import tensorflow as tf

from main import train

flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

flags.DEFINE_string("train", "train", "path to train data")
flags.DEFINE_string("dev", "dev", "path to dev data")
flags.DEFINE_string("test", "train", "path to test data")

flags.DEFINE_string("save_dir", "saved_model/amazon", "path to data dir")
flags.DEFINE_string("data_dir", "data/amazon", "path to data dir")

flags.DEFINE_string("para_log_dir", "log", "directory for log file")
flags.DEFINE_string("log_file", "amazon_hyper_params.txt", "log file name")

flags.DEFINE_string("op_emb", "op_emb", "path to opinion word embedding")
flags.DEFINE_string("word_emb", "ret_emb", "path to word embedding")

flags.DEFINE_integer("max_len", 256, "maximum length in CNN")
flags.DEFINE_integer("num_filters", 100, "number of filters in CNN")
flags.DEFINE_boolean("emb_trainable", False, "whether word embeddings trainable")

flags.DEFINE_integer("batch_size",64, "mini-batch size")
flags.DEFINE_float("keep_prob", 0.7, "dropout rate")
flags.DEFINE_float("learning_rate", 1, "learning rate for adadelta")
flags.DEFINE_float("lr_decay", 0.95, "learning rate decay")
flags.DEFINE_float("l2_reg", 0.0001, "l2 reg for a document classifier")

flags.DEFINE_float("alpha", 0.1, "coefficient of H(q(c|x))")
flags.DEFINE_float("beta", 0.3, "coefficient of regularization")
flags.DEFINE_float("gamma_positive", 0.7, "gamma_1")
flags.DEFINE_float("gamma_negative", -0.1, "gamma_2")

flags.DEFINE_integer("num_epochs", 5, "maximum number of epochs")
flags.DEFINE_integer("num_batches", 200, "number of batches in when evaluating training set not necessary to evaluate all")

flags.DEFINE_string("distance", "euclidean", "distance metric")
flags.DEFINE_string("cos_sim", "min_max", "cosine similairty")

flags.DEFINE_integer("emb_dim", 100, "dimension of word embedding")

flags.DEFINE_integer("eval_period", 100, "evaluate on dev every period")

flags.DEFINE_integer("score_scale", 2, "num of classes")
flags.DEFINE_integer("num_senti", 5, "number of opinion word in sampling")
flags.DEFINE_integer("num_neg", 50, "number of negative opinion word in sampling")
flags.DEFINE_integer("min_count", 1, "min count in batches creation")
flags.DEFINE_boolean("unsupervised", True, "whether to use unsupervised method")

flags.DEFINE_integer("max_to_keep", 1, "number of models to save")
flags.DEFINE_integer("cache_size", 500, "size of dataset buffer")

flags.DEFINE_boolean("verbose", True, "print details or not")


def main(_):
    config = flags.FLAGS
    print('alpha {:.2f} beta {:.2f} gamma_positive {:.2f} gamma_negative {:.2f}'.format(config.alpha, config.beta, config.gamma_positive, config.gamma_negative))
    print('dataset {}'.format(config.data_dir))
    print('unsupervised {} cosine_similarity {}'.format(config.unsupervised, config.cos_sim))
    print('num_filters {} max_len {} emb_dim {} emb_trainable {}'.format(config.num_filters, config.max_len, config.emb_dim, config.emb_trainable))
    print('score_scale {} num_senti {} num_neg {}'.format(config.score_scale, config.num_senti, config.num_neg))
    print('num_epochs {}'.format(config.num_epochs))
    train(config)

if __name__ == "__main__":
    tf.app.run()
