import numpy as np
import tensorflow as tf
from itertools import product
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, roc_curve, auc
from sklearn import metrics

#from munkres import Munkres


class Evaluator:
    def __init__(self):
        self.last_round = None

    def __call__(self, config, model, num_sample, num_batches, sess, handle, str_handle, tag="train", flip=False, verbose=False):
        scale = config.score_scale

        # mean_loss = 0.
        goldens = []
        preds = []

        flag = (num_sample % config.batch_size != 0)
        for i in range(num_batches):
            pred, golden = sess.run([model.pred, model.golden], feed_dict={handle: str_handle})
            # loss, pred, golden = sess.run([model.t_loss, model.pred, model.golden], feed_dict={handle: str_handle})
            # mean_loss += loss
            if (i == num_batches - 1) and flag:
                num_left_over = (num_sample % config.batch_size)
                golden = golden[:num_left_over]  # golden => batch_size x score_scale
                pred = pred[:num_left_over] # pred => batch_size
            golden = np.argmax(golden, axis=1)
            goldens.append(golden)
            preds.append(pred)
        golden = np.concatenate(goldens, axis=0)
        pred = np.concatenate(preds, axis=0)
        # mean_loss = mean_loss / config.num_batches
        # print('len(pred)',len(pred))
        # print('pred[:30]',pred[:30])
        # print('golden[:30]',golden[:30])
        if flag:
            assert (len(golden) == num_sample), "len(golden) {} num_sample {}".format(len(golden),num_sample)
        
        if verbose:
            if len(np.unique(golden)) == 6:
                target_names = ['rec', 'comp', 'sci', 'talk', 'religion', 'misc']
            elif len(np.unique(golden)) == 20:
                target_names = ['rec.autos', 'comp.sys.mac.hardware', 
                                'comp.graphics', 'sci.space', 
                                'talk.politics.guns', 'sci.med',
                                'comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc', 
                                'rec.motorcycles', 'talk.religion.misc',
                                'misc.forsale', 'alt.atheism', 
                                'sci.electronics', 'comp.windows.x', 
                                'rec.sport.hockey', 'rec.sport.baseball', 
                                'soc.religion.christian', 'talk.politics.mideast', 
                                'talk.politics.misc','sci.crypt']
            print(metrics.confusion_matrix(golden, pred))
            print('\n')
            # print(metrics.classification_report(golden, pred, target_names=target_names ,digits=4))
            # print('\n')

        precision = precision_score(golden,pred,average='macro')
        recall = recall_score(golden,pred,average='macro')
        f1_macro = f1_score(golden,pred,average='macro')
        f1_micro = f1_score(golden,pred,average='micro')
        accuracy = accuracy_score(golden, pred)

        # summ = []
        # loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(tag), simple_value=mean_loss), ])
        # overall_acc_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(tag), simple_value=accuracy)])
        # summ.append(loss_sum)
        # summ.append(overall_acc_sum)

        return precision, recall, f1_macro, f1_micro, accuracy, golden, pred

#         if config.unsupervised:
# #             m = Munkres()
#             scale = config.score_scale
#             tots = (golden != -1).sum().astype(np.float32)
#             confusion_mat = np.zeros([scale, scale], dtype=np.int32)
#             for j, k in product(range(scale), range(scale)):
#                 confusion_mat[j, k] = (np.logical_and(golden == j, pred == k).sum())

#             cors = max((confusion_mat[0][0]+confusion_mat[1][1]),(confusion_mat[0][1]+confusion_mat[1][0]))
#             cors = np.asarray(cors, dtype=np.float32)
        # if config.unsupervised:
        #     scale = config.score_scale
        #     tots = (golden != -1).sum().astype(np.float32)
        #     confusion_mat = np.zeros([scale, scale], dtype=np.int32)
        #     for j, k in product(range(scale), range(scale)):
        #         confusion_mat[j, k] = (np.logical_and(golden == j, pred == k).sum())
            
        #     if (confusion_mat[0][0]+confusion_mat[1][1]) > (confusion_mat[0][1]+confusion_mat[1][0]):
        #         is_flip = False
        #     else:
        #         is_flip = True 
        #     cors = max((confusion_mat[0][0]+confusion_mat[1][1]),(confusion_mat[0][1]+confusion_mat[1][0]))
        #     cors = np.asarray(cors, dtype=np.float32)
        #     if is_flip:
        #         pred = 1 - pred

        # else:
        #     tots = (golden != -1).sum().astype(np.float32)
        #     cors = (golden == pred).sum().astype(np.float32)
        #     is_flip = False
        
        
        
        # print('precision',precision,'recall',recall,'f1_macro',f1_macro,'f1_micro',f1_micro,'accuracy',accuracy) 

        
        

        # if flip and config.unsupervised:
        #     self.last_round = pred if self.last_round is None else self.last_round
        #     flip = (pred != self.last_round).sum()
        #     self.last_round = pred
        #     flip_summ = tf.Summary(value=[tf.Summary.Value(tag="{}/flip".format(tag), simple_value=flip)])
        #     summ.append(flip_summ)

        
