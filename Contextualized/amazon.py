import os
import argparse
from main import train

parser = argparse.ArgumentParser(description='VWS')

parser.add_argument('--unsupervised', type=bool, default=True,
                    help='whether train model in an unsupervised way')

parser.add_argument('--train', type=str, default="train",
                    help='training set')
parser.add_argument('--dev', type=str, default="dev",
                    help='development set')
parser.add_argument('--test', type=str, default="train",
                    help='test set')

parser.add_argument('--num_chunk', type=int, default=5,
                    help='dissimilarity threshold')

parser.add_argument('--save_dir', type=str, default="saved_model/amazon",
                    help='path to model saving directory')
parser.add_argument('--data_dir', type=str, default="../data/amazon",
                    help='path to data directory')
parser.add_argument('--log_dir', type=str, default="log/amazon",
                    help='path to log directory')
parser.add_argument('--log_file', type=str, default="amazon_hyper_params.txt",
                    help='file to record f1 score')

parser.add_argument('--op_emb', type=str, default="op_emb",
                    help='path to opinion word embedding')
parser.add_argument('--word_emb', type=str, default="ret_emb",
                    help='path to word embedding')

parser.add_argument('--alpha', type=float, default=0.1,
                    help='coefficient of H(q(c|x))')
parser.add_argument('--beta', type=float, default=0.1,
                    help='coefficient of regularization')


parser.add_argument('--gamma_positive', type=float, default=0.7,
                    help='similarity threshold: gamma_1')
parser.add_argument('--gamma_negative', type=float, default=-0.1,
                    help='dissimilarity threshold: gamma_2')
parser.add_argument('--score_scale', type=int, default=2,
                    help='num classes')
parser.add_argument('--num_senti', type=int, default=5,
                    help='number of opinion words sampled in a document')
parser.add_argument('--num_neg', type=int, default=50,
                    help='number of negative opinion words sampled in a document')
parser.add_argument('--min_count', type=int, default=1,
                    help='words that less than min_count will be filtered out in opinion word classifier')

parser.add_argument('--max_len', type=int, default=128,
                    help='maximum document length in CNN')
parser.add_argument('--num_filters', type=int, default=100,
                    help='number of filters in CNN')
parser.add_argument('--emb_dim', type=int, default=100,
                    help='dimension of embedding in CNN')
parser.add_argument('--emb_trainable', type=bool, default=False,
                    help='whether word embeddings is trainable')

parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=5,
                    help='maximum number of epochs')
parser.add_argument('--eval_period', type=int, default=100,
                    help='evaluate on dev every period')

parser.add_argument('--keep_prob', type=float, default=0.7,
                    help='keep probability in dropout')
parser.add_argument('--lr', type=float, default=1,
                    help='learning rate for Adadelta')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='learning rate decay')
parser.add_argument('--l2_reg', type=float, default=0.0001,
                    help='l2 regularization of the sentiment classifier')

parser.add_argument('--verbose', type=bool, default=True,
                    help='whether print details')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='whether use cuda')

args = parser.parse_args()

  
print('alpha {:.2f} beta {:.2f} gamma_positive {:.2f} gamma_negative {:.2f}'.format(args.alpha, args.beta, args.gamma_positive, args.gamma_negative))
print('dataset {}'.format(args.data_dir))
print('unsupervised {}'.format(args.unsupervised))
print('num_filters {} max_len {} emb_dim {} emb_trainable {}'.format(args.num_filters, args.max_len, args.emb_dim, args.emb_trainable))
print('score_scale {} num_senti {} num_neg {}'.format(args.score_scale, args.num_senti, args.num_neg))
print('num_epochs {}'.format(args.num_epochs))
train(args)
