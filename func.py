import tensorflow as tf


def dropout(args, keep_prob, is_train):
    if keep_prob < 1.0:
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob), lambda: args)
    return args


def linear(inputs, W):
    shape = tf.shape(inputs)
    dim = inputs.get_shape().as_list()[-1]
    out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [W.get_shape().as_list()[-1]]
    flat_inputs = tf.reshape(inputs, [-1, dim])
    res = tf.matmul(flat_inputs, W)
    res = tf.reshape(res, out_shape)
    return res


def selectional_preference(sent_emb, neg_sent_emb, negation, probs, score_scale, scale=0.1):
    """
    calculate E_{q(x)} log [sigmoid(u)] + log [sigmoid(-u_neg_sample)]
    :param sent_emb: batch_size x num_senti 
    :param neg_sent_emb: batch_size x num_senti 
    :param name: negation
    """

    with tf.variable_scope("selectional_preference"):
        emb_dim = sent_emb.get_shape().as_list()[-1]

        W = tf.get_variable("W", [emb_dim, score_scale])
        W_norm = tf.norm(W, axis=0, keepdims=True)
        W = W / W_norm * tf.get_variable("scale", [], initializer=tf.constant_initializer(scale))

        negation_ = tf.expand_dims(negation, axis=2) # negation_ => batch_size x num_senti_ x 1
        u = tf.nn.sigmoid(linear(sent_emb, W) * negation_) # u => batch_size x num_senti x score_scale
        u_neg_sample = tf.nn.sigmoid(-linear(neg_sent_emb, W)) # u_neg_sample => batch_size x num_senti x score_scale

        log_u = tf.reduce_mean(tf.log(u), axis=1) # log_u => batch_size x score_scale
        log_u_neg_sample = tf.reduce_mean(tf.log(u_neg_sample), axis=1) # log_u_neg_sample => batch_size x score_scale

        entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.log(probs),axis=1))
        vae_loss = tf.reduce_mean(tf.reduce_sum((log_u + log_u_neg_sample )*probs, axis=1))

        return -vae_loss, -entropy, W, u, u_neg_sample, log_u, log_u_neg_sample

def tf_cal_kl(q, p, name=None):
    """
    calculate KL(q||p) = \sum_r q \log(q/p)
    :param q: (...,r)
    :param p: (..., r) same dim with q
    :param name: op name
    :return: kl, (...,)
    """
    p = tf.maximum(p, 1e-6)
    kl = tf.reduce_sum(q * tf.log(q / p), axis=-1, name=name)
    return kl


def block_wise_operator(mat,a,b):
    mat = tf.reshape(mat,[a,b,-1])
    mat = tf.transpose(mat,[0,2,1])
    mat = tf.reshape(mat,[a,a,b,b])
    mat = tf.transpose(mat,[0,1,3,2])
    return mat


def get_cos_min_max_similarity(config, senti_emb, negation):
    negation_ = tf.tile(tf.expand_dims(negation, axis=2),[1, 1, tf.shape(senti_emb)[-1]])
    senti_emb = senti_emb*negation_

    senti_emb = tf.reshape(senti_emb, [-1, tf.shape(senti_emb)[-1]])
    senti_emb = tf.nn.l2_normalize(senti_emb, axis=1)
    
    bn_x_bn_cos_sim = tf.matmul(senti_emb, tf.transpose(senti_emb, [1, 0]), name='enlarge_cos_similarity')
    bn_x_bn_cos_sim = block_wise_operator(bn_x_bn_cos_sim,config.batch_size,config.num_senti)
    
    
    b_x_b_max = tf.reduce_max(bn_x_bn_cos_sim,axis=[2,3])
    b_x_b_min = tf.reduce_min(bn_x_bn_cos_sim,axis=[2,3])
    

    max_mask = tf.math.greater(b_x_b_max, config.gamma_positive)
    min_mask = tf.math.less(b_x_b_min, config.gamma_negative)


    and_mask = tf.cast(tf.math.logical_and(min_mask,max_mask),dtype=tf.float32)
    and_mask_complement = tf.constant([1.0]) - and_mask
    
    max_mask = tf.cast(max_mask,dtype=tf.float32)
    min_mask = tf.cast(min_mask,dtype=tf.float32)  
    
    max_mask = tf.matrix_set_diag(max_mask, tf.zeros((tf.shape(max_mask)[0],), dtype=tf.float32))
    min_mask = tf.matrix_set_diag(min_mask, tf.zeros((tf.shape(min_mask)[0],), dtype=tf.float32))
    
    and_mask_complement = tf.matrix_set_diag(and_mask_complement, tf.zeros((tf.shape(and_mask_complement)[0],), dtype=tf.float32))
    and_mask = tf.matrix_set_diag(and_mask, tf.zeros((tf.shape(and_mask)[0],), dtype=tf.float32))
    
    similarity = b_x_b_max*and_mask_complement*max_mask + b_x_b_min*and_mask_complement*min_mask + and_mask
    return similarity, (and_mask_complement*max_mask+and_mask_complement*min_mask+and_mask), and_mask, b_x_b_min, b_x_b_max
    

def get_distance(config, probs):
    # b, 1, num_class ->  b, b, num_class
    r_p_matrix = tf.tile(tf.expand_dims(probs, axis=1),
                        [1, tf.shape(probs)[0], 1])
    r_p_matrix_T = tf.transpose(r_p_matrix, [1, 0, 2])
    if config.distance == 'euclidean':
        #'Prob(R) distance: Euclidean'
        r_p_matrix_distance = tf.squared_difference(r_p_matrix, r_p_matrix_T)
        epsilon = 1e-12
        r_p_distance = tf.sqrt(tf.maximum(tf.reduce_sum(r_p_matrix_distance, 2), epsilon), name='euclidean_distance')
    elif config.distance == 'JS':
        #'Prob(R) distance: Jensen-Shannon'
        average = (r_p_matrix + r_p_matrix_T) / 2
        r_p_distance = tf.divide(
            tf_cal_kl(r_p_matrix, average) + tf_cal_kl(r_p_matrix_T, average), 
            2., name='JS_divergence')
    elif config.distance == 'KL':
        #'Prob(R) distance: KL'
        r_p_distance = tf_cal_kl(r_p_matrix, r_p_matrix_T, name='KL_divergence')
    else:
        raise NotImplementedError()
    return r_p_distance


def get_regularizer_score_pairwise(config, probs, senti_emb, negation):
    """
    sent_emb b x num_senti x embd_size
    negation b x num_senti x embd_size
    probs b x num_class

    """
    r_p_distance = get_distance(config, probs)  
    
    similarity, mask, b_x_b_mean, b_x_b_min, b_x_b_max = get_cos_min_max_similarity(config, senti_emb, negation)
    
    pair_cnt = tf.reduce_sum(mask)

    regularizer_score = tf.truediv(tf.reduce_sum(r_p_distance * similarity),(pair_cnt + tf.constant(1e-6)))
    
    return regularizer_score, similarity, b_x_b_mean, b_x_b_min, b_x_b_max

