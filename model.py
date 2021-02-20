import tensorflow as tf
from func import selectional_preference
from func import get_regularizer_score_pairwise
from func import dropout

class Model:
    def __init__(self, config, batch, word_mat, op_word_mat):
        self.config = config
        self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.get_variable("lr", [], initializer=tf.constant_initializer(config.learning_rate), trainable=False)
        self.x, self.y, self.senti, self.neg_senti, self.negation = batch.get_next()
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),trainable=config.emb_trainable)
        self.op_word_mat = tf.get_variable("op_word_mat", initializer=tf.constant(op_word_mat, dtype=tf.float32))
        
        self.alpha = tf.get_variable("alpha",[],initializer=tf.constant_initializer(config.alpha),trainable=False)
        self.beta = tf.get_variable("beta",[],initializer=tf.constant_initializer(config.beta),trainable=False)
        
        self.loss, self.total_loss = None, None
        self.vae_loss, self.entropy_term_loss, self.opinion_reg_loss = None, None, None
        
        self.filter_sizes = [2,3,4,5]
        self.num_filters = config.num_filters
        self.emb_dim = config.emb_dim
        self.senti_emb = None

        self.similarity, self.r_p_distance, self.b_x_b_mean, self.b_x_b_min, self.b_x_b_min = None, None, None, None, None
        self.dot_product, self.dot_product_idx = None, None
        self.ready()

        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
        pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="predict")
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
        self.var_to_save = enc_vars + pred_vars
        # for var in self.var_to_save:
        #     print('var to save',var)
        reg = tf.contrib.layers.l2_regularizer(config.l2_reg)
        enc_l2_loss = tf.contrib.layers.apply_regularization(reg, enc_vars)
        pred_l2_loss = tf.contrib.layers.apply_regularization(reg, pred_vars)
        dec_l2_loss = tf.contrib.layers.apply_regularization(reg, dec_vars)
        
        if config.unsupervised:
            self.total_loss = self.vae_loss + self.entropy_term_loss + self.opinion_reg_loss \
            + enc_l2_loss + dec_l2_loss + pred_l2_loss #+ self.vocab_reg_loss + self.opinion_reg_loss  + self.opinion_reg_loss
            var_list = enc_vars + pred_vars + dec_vars + [self.op_word_mat]
        else:
            self.total_loss =  self.loss + enc_l2_loss + pred_l2_loss
            var_list = enc_vars + pred_vars
            if config.emb_trainable:
                var_list += [self.word_mat]
        
        for var in var_list:
            print(var)
        
        self.opt = tf.train.AdadeltaOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.total_loss, global_step=self.global_step, var_list=var_list)
    
  
    def ready(self):
        config = self.config
        x, senti, neg_senti, negation = self.x, self.senti, self.neg_senti, self.negation
        
        word_mat, op_word_mat = self.word_mat, self.op_word_mat

        score_scale = config.score_scale
        
        with tf.variable_scope("encoder"):
            x = tf.nn.embedding_lookup(word_mat,x)
            x = tf.expand_dims(x,-1)
            x = dropout(x,keep_prob=config.keep_prob,is_train=self.is_train)
            
            pooled_outputs = []
            for f_size in self.filter_sizes:
                conv = tf.layers.conv2d(
                x, 
                filters=self.num_filters,
                kernel_size=[f_size, self.emb_dim],
                strides=(1,1),
                padding='VALID',
                activation=tf.nn.relu)

                pool = tf.layers.max_pooling2d(
                conv,
                pool_size=[config.max_len-f_size + 1, 1],
                strides=(1,1),
                padding='VALID')

                pooled_outputs.append(pool)

            h_pool = tf.concat(pooled_outputs,3)
            h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters*len(self.filter_sizes)])
            h_drop = dropout(h_pool_flat, config.keep_prob,is_train=self.is_train)
    
        with tf.variable_scope("predict"):
            logit = tf.layers.dense(h_drop, config.score_scale, activation=None)
            self.prob = tf.nn.softmax(logit)
            self.pred = tf.argmax(self.prob,axis=-1)
            self.golden = self.y
            self.loss = tf.reduce_mean(tf.reduce_sum(-self.golden * tf.log(self.prob + 1e-6), axis=1))

        with tf.variable_scope("decoder"):
            senti_emb = tf.nn.embedding_lookup(op_word_mat, senti)
            self.senti_emb = senti_emb

            neg_senti_emb = tf.nn.embedding_lookup(op_word_mat, neg_senti)
            self.neg_senti_emb = neg_senti_emb
            
            self.vae_loss, entropy_term, self.W_decoder, self.u, self.u_neg_sample, self.log_u, self.log_u_neg_sample = selectional_preference(senti_emb, neg_senti_emb, negation, self.prob, score_scale)
            self.entropy_term_loss = tf.multiply(self.alpha, entropy_term, name="entropy_term")
            
            opinion_reg, self.similarity, self.b_x_b_mean, self.b_x_b_min, self.b_x_b_max = get_regularizer_score_pairwise(config, self.prob, senti_emb, negation)
            self.opinion_reg_loss = tf.multiply(self.beta, opinion_reg, name="opinion_words_regulazation")



