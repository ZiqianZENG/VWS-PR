import os
import random
from itertools import chain
import tensorflow as tf
from model import Model
from util.batch_gen import batch_generator, list_wrapper
from util.load import load_corpus, load_embedding
from tqdm import tqdm
from evaluator import Evaluator
import numpy as np

def train(config):
    op_word2idx, op_emb = load_embedding(config, os.path.join(config.data_dir,config.op_emb),True)
    word2idx, word_emb = load_embedding(config, os.path.join(config.data_dir,config.word_emb))

    op_idx2word = { op_word2idx[word]:word for word in op_word2idx }
    op_idx2word[0] = 'NULL'
    print("Building Batches")
    
    need_to_generate_neg_senti = config.unsupervised
    
    train_corpus = load_corpus(config, os.path.join(config.data_dir,config.train), 'train', word2idx, op_word2idx, filter_null=config.unsupervised)
    num_train_sample = len(train_corpus[0])
    train_batch_list = list(batch_generator(config, True, train_corpus, need_to_generate_neg_senti))

    dev_corpus = load_corpus(config, os.path.join(config.data_dir,config.dev), 'dev', word2idx, op_word2idx)
    num_dev_sample = len(dev_corpus[0])
    dev_batch_list = list(batch_generator(config, False, dev_corpus, False))

    test_corpus = load_corpus(config, os.path.join(config.data_dir,config.test), 'test', word2idx, op_word2idx)
    num_test_sample = len(test_corpus[0])
    test_batch_list = list(batch_generator(config, False, test_corpus, False))

    random.shuffle(train_batch_list)

    num_train_batch = len(train_batch_list)
    num_dev_batch = len(dev_batch_list)
    num_test_batch = len(test_batch_list)

    input_types = (tf.int32, tf.float32, tf.int32, tf.int32, tf.float32)
    input_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]))
    
    train_batch = tf.data.Dataset.from_generator(list_wrapper(
        train_batch_list), input_types, input_shapes).repeat().shuffle(config.cache_size).make_one_shot_iterator()
    dev_batch = tf.data.Dataset.from_generator(list_wrapper(
        dev_batch_list), input_types, input_shapes).repeat().make_one_shot_iterator()
    test_batch = tf.data.Dataset.from_generator(list_wrapper(
        test_batch_list), input_types, input_shapes).repeat().make_one_shot_iterator()

    train_evaluator = Evaluator()
    dev_evaluator = Evaluator()
    test_evaluator = Evaluator()

    handle = tf.placeholder(tf.string, shape=[])
    batch = tf.data.Iterator.from_string_handle(handle, train_batch.output_types, train_batch.output_shapes)

    model = Model(config, batch, word_emb, op_emb)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        if config.verbose:
            writer = tf.summary.FileWriter(config.para_log_dir)
        sess.run(tf.global_variables_initializer())
        train_handle = sess.run(train_batch.string_handle())
        dev_handle = sess.run(dev_batch.string_handle())
        test_handle = sess.run(test_batch.string_handle())

        saver = tf.train.Saver(var_list=model.var_to_save, max_to_keep=config.max_to_keep)
        if config.unsupervised:
            sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            dev_precision, dev_recall, dev_f1_macro, dev_f1_micro, dev_accuracy, dev_golden, dev_pred  = \
                dev_evaluator(config, model, num_dev_sample, num_dev_batch, sess, handle, dev_handle, tag="dev", flip=True, verbose=False)
                
            test_precision, test_recall, test_f1_macro, test_f1_micro, test_accuracy, test_golden, test_pred = \
                test_evaluator(config, model, num_test_sample, num_test_batch, sess, handle, test_handle, tag="test", flip=True)

            print('Pretraining Results:')
            print('Dev\n')
            print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(dev_precision, dev_recall, dev_f1_micro, dev_f1_macro, dev_accuracy))

            print('Test\n')
            print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(test_precision, test_recall, test_f1_micro, test_f1_macro,test_accuracy)) 
            

        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        best_dev_precision, best_dev_recall, best_dev_f1_macro, best_dev_f1_micro, best_dev_accuracy = 0., 0., 0., 0., 0.
        best_test_precision, best_test_recall, best_dev_f1_macro, best_test_f1_micro, best_test_accuracy = 0., 0., 0., 0., 0.
        best_epoch = 0
        for epoch in tqdm(range(1, num_train_batch * config.num_epochs + 1)): # range(1, num_train_batch * config.num_epochs + 1):
            global_step = sess.run(model.global_step) + 1
            
            
            vae_loss, entropy_term_loss, opinion_reg_loss, _ = sess.run(
                    [model.vae_loss, model.entropy_term_loss, model.opinion_reg_loss, model.train_op], feed_dict={handle: train_handle})
       
            # model_W_decoder, model_u, model_u_neg_sample, model_log_u, model_log_u_neg_sample, _ = sess.run(
            #     [model.vae_loss, model.entropy_term_loss, model.opinion_reg_loss, model.prob, model.and_mask, model.b_x_b_min, model.b_x_b_max, model.senti, model.y, \
            #     model.W_decoder, model.u, model.u_neg_sample, model.log_u, model.log_u_neg_sample, model.train_op], feed_dict={handle: train_handle})

            # if global_step % config.record_period == 0:
            #     if config.verbose and config.unsupervised:
            #         loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/vae_loss", simple_value=vae_loss), ])
            #         writer.add_summary(loss_sum, global_step)
            #         writer.flush()
            #         reg_loss_summ = tf.Summary(value=[tf.Summary.Value(tag="model/entropy_term_loss", simple_value=entropy_term_loss), ])
            #         writer.add_summary(reg_loss_summ, global_step)
            #         writer.flush()
            #     else:
            #         pass

            if global_step % config.eval_period == 0:
                sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
                # _, _, train_summ, _, _, _, _, _, _,_,_,is_flip= train_evaluator(config, model, config.num_batches, sess, handle, train_handle, tag="train")
                dev_precision, dev_recall, dev_f1_macro, dev_f1_micro, dev_accuracy, dev_golden, dev_pred  = \
                dev_evaluator(config, model, num_dev_sample, num_dev_batch, sess, handle, dev_handle, tag="dev", flip=True, verbose=True)
                
                test_precision, test_recall, test_f1_macro, test_f1_micro, test_accuracy, test_golden, test_pred = \
                test_evaluator(config, model, num_test_sample, num_test_batch, sess, handle, test_handle, tag="test", flip=True)
                
                # if config.verbose:
                #     for s in chain(train_summ, dev_summ, test_summ):
                #         writer.add_summary(s, global_step)
                #     writer.flush()

                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))

                if dev_f1_micro > best_dev_f1_micro:
                    best_epoch = epoch
                    best_dev_precision, best_dev_recall, best_dev_f1_macro, best_dev_f1_micro, best_dev_accuracy = dev_precision, dev_recall, dev_f1_macro, dev_f1_micro, dev_accuracy
                    best_test_precision, best_test_recall, best_test_f1_macro, best_test_f1_micro, best_test_accuracy = test_precision, test_recall, test_f1_macro, test_f1_micro, test_accuracy 
                    if not config.unsupervised:
                        filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                        saver.save(sess, filename)
                if config.verbose:
                    print('\nalpha {:.4f} beta {:.4f} gamma_positive {:.4f} gamma_negative {:.4f}'.format(config.alpha, config.beta, config.gamma_positive, config.gamma_negative))
                    print('dataset {}'.format(config.data_dir))
                    print('unsupervised {}'.format(config.unsupervised))
                    print('num_filters {} max_len {} emb_dim {} emb_trainable {}'.format(config.num_filters, config.max_len, config.emb_dim, config.emb_trainable))
                    print('score_scale {} num_senti {} num_neg {}'.format(config.score_scale, config.num_senti, config.num_neg))
                    print('num_epochs {}'.format(config.num_epochs))
                    
                    print('Step {}\n'.format(global_step))
                    print('Dev\n')
                    print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(dev_precision, dev_recall, dev_f1_micro, dev_f1_macro, dev_accuracy))

                    print('Test\n')
                    print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(test_precision, test_recall, test_f1_micro, test_f1_macro,test_accuracy)) 
                
                if epoch >= 10:
                    sess.run(tf.assign(model.lr, model.lr * config.lr_decay))

        if config.verbose:
            print('\nalpha {:.4f} beta {:.4f} gamma_positive {:.4f} gamma_negative {:.4f}'.format(config.alpha, config.beta, config.gamma_positive, config.gamma_negative))
            print('dataset {}'.format(config.data_dir))
            print('unsupervised {}'.format(config.unsupervised))
            print('num_filters {} max_len {} emb_dim {} emb_trainable {}'.format(config.num_filters, config.max_len, config.emb_dim, config.emb_trainable))
            print('score_scale {} num_senti {} num_neg {}'.format(config.score_scale, config.num_senti, config.num_neg))
            print('num_epochs {}'.format(config.num_epochs))

            print('Best Epoch at {:2}'.format(best_epoch))
            print('Dev\n')
            print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(best_dev_precision, best_dev_recall, best_dev_f1_micro, best_dev_f1_macro, best_dev_accuracy))

            print('Test\n')
            print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(best_test_precision, best_test_recall, best_test_f1_micro, best_test_f1_macro, best_test_accuracy)) 


        if not os.path.exists(config.para_log_dir):
            os.makedirs(config.para_log_dir)
            if not os.path.exists(os.path.join(config.para_log_dir,config.log_file)):
                with open(os.path.join(config.para_log_dir,config.log_file),'w') as fo:
                    pass
                

        with open(os.path.join(config.para_log_dir,config.log_file),'a') as fo:
            fo.write('alpha {:.4f} beta {:.4f} gamma_positive {:.4f} gamma_negative {:.4f}\n'.format(
                config.alpha, config.beta, config.gamma_positive, config.gamma_negative))
            fo.write('dataset {}\n'.format(config.data_dir))
            fo.write('unsupervised {} \n'.format(config.unsupervised))
            fo.write('num_filters {} max_len {} emb_dim {} emb_trainable {} \n'.format(config.num_filters, config.max_len, config.emb_dim, config.emb_trainable))
            fo.write('score_scale {} num_senti {} num_neg {} \n'.format(config.score_scale, config.num_senti, config.num_neg))
            fo.write('num_epochs {} \n'.format(config.num_epochs))
            fo.write('Best Epoch at {}\n'.format(best_epoch))
            fo.write('Dev\n')
            fo.write('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(best_dev_precision, best_dev_recall, best_dev_f1_micro, best_dev_f1_macro, best_dev_accuracy))

            fo.write('Test\n')
            fo.write('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(best_test_precision, best_test_recall, best_test_f1_micro, best_test_f1_macro, best_test_accuracy))

            fo.write('='*100 + '\n')
        
 
