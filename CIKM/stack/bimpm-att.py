
# coding: utf-8

# In[1]:


# For limiting the GPU memory usage
import os
import seaborn as sb
GPU_ID = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import keras
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.cluster import DBSCAN
import networkx as nx
import shutil

import numpy as np
import os, json, re
from collections import Counter

import tensorflow as tf
from keras import backend as K
import pickle as pk
from keras.layers import *
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras.layers.merge import Concatenate, Average
import numpy as np
from sklearn import metrics
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from gensim.models import Word2Vec

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
import gensim
from keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.layers.pooling import GlobalAveragePooling1D
from keras import backend as K
import codecs
from sklearn.metrics import classification_report, f1_score
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


# In[2]:


def log_loss(y_test, pred):
    loss = []
    for i, j in zip(pred, y_test):
        if j == 1:
            loss += [np.log(i)]
        else:
            loss += [np.log(1-i)]
    return (loss)
en1 = pk.load(open('../data/en1.dat'))
en2 = pk.load(open('../data/en2.dat'))
enu = pk.load(open('../data/enu.dat'))
lab = pk.load(open('../data/out.dat'))

es1 = pk.load(open('../data/es1.dat'))
es2 = pk.load(open('../data/es2.dat'))
esu = pk.load(open('../data/esu.dat'))

en2vec = pk.load(open('../data/en2vec.dat'))
es2vec = pk.load(open('../data/es2vec.dat'))
test1, test2 = pk.load(open('../data/test.dat'))

gt = pk.load(open('../data/groundtruth.txt'))
gt = np.array(gt)

en = list(en1) + list(en2) + list(enu)
es = list(es1) + list(es2) + list(esu) + list(test1) + list(test2)

es_aug1 = pk.load(open('../data/es_aug1.dat'))
es_aug2 = pk.load(open('../data/es_aug2.dat'))

es2id = pk.load(open('../data/es2id.dat'))
id2es = {es2id[i] : i for i in es2id}


# In[3]:


class MultiPerspective(Layer):
    """Multi-perspective Matching Layer.
    # Arguments
        mp_dim: single forward/backward multi-perspective dimention
    """

    def __init__(self, mp_dim, epsilon=1e-6, **kwargs):
        self.mp_dim = mp_dim
        self.epsilon = 1e-6
        self.strategy = 4
        super(MultiPerspective, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        embedding_size = input_shape[-1] / 2
        # Create a trainable weight variable for this layer.
        # input_shape is bidirectional RNN input shape
        # kernel shape (mp_dim * 2 * self.strategy, embedding_size)
        self.kernel = self.add_weight((self.mp_dim,
                                       embedding_size * 2 * self.strategy),
                                       name='kernel',
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.kernel_full_fw = self.kernel[:, :embedding_size]
        self.kernel_full_bw = self.kernel[:, embedding_size: embedding_size * 2]
        self.kernel_attentive_fw = self.kernel[:, embedding_size * 2: embedding_size * 3]
        self.kernel_attentive_bw = self.kernel[:, embedding_size * 3: embedding_size * 4]
        self.kernel_max_attentive_fw = self.kernel[:, embedding_size * 4: embedding_size * 5]
        self.kernel_max_attentive_bw = self.kernel[:, embedding_size * 5: embedding_size * 6]
        self.kernel_max_pool_fw = self.kernel[:, embedding_size * 6: embedding_size * 7]
        self.kernel_max_pool_bw = self.kernel[:, embedding_size * 7:]
        self.built = True
        super(MultiPerspective, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], self.mp_dim * 2 * self.strategy)

    def get_config(self):
        config = {'mp_dim': self.mp_dim,
                  'epsilon': self.epsilon}
        base_config = super(MultiPerspective, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        # h1, h2: bidirectional LSTM hidden states, include forward and backward states
        #         (batch_size, timesteps, embedding_size * 2)
        h1 = inputs[0]
        h2 = inputs[1]
        embedding_size = K.int_shape(h1)[-1] / 2
        h1_fw = h1[:, :, :embedding_size]
        h1_bw = h1[:, :, embedding_size:]
        h2_fw = h2[:, :, :embedding_size]
        h2_bw = h2[:, :, embedding_size:]

        # 4 matching strategy
        list_matching = []

        # full matching ops
        matching_fw = self._full_matching(h1_fw, h2_fw, self.kernel_full_fw)
        matching_bw = self._full_matching(h1_bw, h2_bw, self.kernel_full_bw)
        list_matching.extend([matching_fw, matching_bw])

        
        h1_fw, mask1 = self._mask(h1_fw)
        h2_fw, mask2 = self._mask(h2_fw)
        
        # cosine matrix
        cosine_matrix_fw = self._cosine_matrix(h1_fw, h2_fw, mask1, mask2)
        
        h1_bw, mask1 = self._mask(h1_fw)
        h2_bw, mask2 = self._mask(h2_fw)
        cosine_matrix_bw = self._cosine_matrix(h1_bw, h2_bw, mask1, mask2)

        # attentive matching ops
        matching_fw = self._attentive_matching(
            h1_fw, h2_fw, cosine_matrix_fw, self.kernel_attentive_fw)
        matching_bw = self._attentive_matching(
            h1_bw, h2_bw, cosine_matrix_bw, self.kernel_attentive_bw)
        list_matching.extend([matching_fw, matching_bw])

        # max attentive matching ops
        matching_fw = self._max_attentive_matching(
            h1_fw, h2_fw, cosine_matrix_fw, self.kernel_max_attentive_fw)
        matching_bw = self._max_attentive_matching(
            h1_bw, h2_bw, cosine_matrix_bw, self.kernel_max_attentive_bw)
        list_matching.extend([matching_fw, matching_bw])

        # max pooling matching ops
        matching_fw = self._max_pooling_matching(h1_fw, h2_fw, self.kernel_max_pool_fw)
        matching_bw = self._max_pooling_matching(h1_bw, h2_bw, self.kernel_max_pool_bw)
        list_matching.extend([matching_fw, matching_bw])

        return K.concatenate(list_matching, axis=-1)

    def _mask(self, x1):
        h1 = K.expand_dims(x1, axis=2)
        # expand x2 shape to (batch_size, 1, x2_timesteps, embedding_size)
        h2 = K.expand_dims(x1, axis=1)
        cos_matrix = self._cosine_similarity(h1, h2)
        mask = K.cast(cos_matrix > 1-self.epsilon, np.float32)
        bool_mask = 1-K.cast(K.sum(mask, axis=1) > 1, np.float32)
        return x1, bool_mask    
    
    def _cosine_similarity(self, x1, x2):
        """Compute cosine similarity.
        # Arguments:
            x1: (..., embedding_size)
            x2: (..., embedding_size)
        """
        cos = K.sum(x1 * x2, axis=-1)
        x1_norm = K.sqrt(K.maximum(K.sum(K.square(x1), axis=-1), self.epsilon))
        x2_norm = K.sqrt(K.maximum(K.sum(K.square(x2), axis=-1), self.epsilon))
        cos = cos / x1_norm / x2_norm
        return cos


    def _cosine_matrix(self, x1, x2, mask1, mask2):
        """Cosine similarity matrix.
        Calculate the cosine similarities between each forward (or backward)
        contextual embedding h_i_p and every forward (or backward)
        contextual embeddings of the other sentence
        # Arguments
            x1: (batch_size, x1_timesteps, embedding_size)
            x2: (batch_size, x2_timesteps, embedding_size)
        # Output shape
            (batch_size, x1_timesteps, x2_timesteps)
        """
        # expand h1 shape to (batch_size, x1_timesteps, 1, embedding_size)
        h1 = K.expand_dims(x1, axis=2)
        # expand x2 shape to (batch_size, 1, x2_timesteps, embedding_size)
        h2 = K.expand_dims(x2, axis=1)
        # cosine matrix (batch_size, h1_timesteps, h2_timesteps)
        
        
        
        cos_matrix = K.exp(self._cosine_similarity(h1, h2))

        line = x1.shape.as_list()[1]
        if type(line) != int:
            line = 48
        
        mask1 = K.expand_dims(mask1, axis=2)
        mask1 = K.repeat_elements(mask1, rep=line, axis=2)
        cos_matrix = cos_matrix * mask1
        
        
        mask2 = K.expand_dims(mask2, axis=1)
        mask2 = K.repeat_elements(mask2, rep=line, axis=1)
        cos_matrix = cos_matrix * mask2
        
        return cos_matrix
        

    def _mean_attentive_vectors(self, x2, cosine_matrix):
        """Mean attentive vectors.
        Calculate mean attentive vector for the entire sentence by weighted
        summing all the contextual embeddings of the entire sentence
        # Arguments
            x2: sequence vectors, (batch_size, x2_timesteps, embedding_size)
            cosine_matrix: cosine similarities matrix of x1 and x2,
                           (batch_size, x1_timesteps, x2_timesteps)
        # Output shape
            (batch_size, x1_timesteps, embedding_size)
        """
        # (batch_size, x1_timesteps, x2_timesteps, 1)
        expanded_cosine_matrix = K.expand_dims(cosine_matrix, axis=-1)
        # (batch_size, 1, x2_timesteps, embedding_size)
        x2 = K.expand_dims(x2, axis=1)
        # (batch_size, x1_timesteps, embedding_size)
        weighted_sum = K.sum(expanded_cosine_matrix * x2, axis=1)
        # (batch_size, x1_timesteps, 1)
        sum_cosine = K.expand_dims(K.sum(cosine_matrix, axis=-1) + self.epsilon, axis=-1)
        # (batch_size, x1_timesteps, embedding_size)
        attentive_vector = weighted_sum / sum_cosine
        return attentive_vector

    def _max_attentive_vectors(self, x2, cosine_matrix):
        """Max attentive vectors.
        Calculate max attentive vector for the entire sentence by picking
        the contextual embedding with the highest cosine similarity
        as the attentive vector.
        # Arguments
            x2: sequence vectors, (batch_size, x2_timesteps, embedding_size)
            cosine_matrix: cosine similarities matrix of x1 and x2,
                           (batch_size, x1_timesteps, x2_timesteps)
        # Output shape
            (batch_size, x1_timesteps, embedding_size)
        """
        # (batch_size, x1_timesteps)
        max_x2_step = K.argmax(cosine_matrix, axis=-1)

        embedding_size = K.int_shape(x2)[-1]
        timesteps = K.int_shape(max_x2_step)[-1]
        if timesteps is None:
            timesteps = K.shape(max_x2_step)[-1]

        # collapse time dimension and batch dimension together
        # collapse x2 to (batch_size * x2_timestep, embedding_size)
        x2 = K.reshape(x2, (-1, embedding_size))
        # collapse max_x2_step to (batch_size * h1_timesteps)
        max_x2_step = K.reshape(max_x2_step, (-1,))
        # (batch_size * x1_timesteps, embedding_size)
        max_x2 = K.gather(x2, max_x2_step)
        # reshape max_x2, (batch_size, x1_timesteps, embedding_size)
        attentive_vector = K.reshape(max_x2, K.stack([-1, timesteps, embedding_size]))
        return attentive_vector

    def _time_distributed_multiply(self, x, w):
        """Element-wise multiply vector and weights.
        # Arguments
            x: sequence of hidden states, (batch_size, ?, embedding_size)
            w: weights of one matching strategy of one direction,
               (mp_dim, embedding_size)
        # Output shape
            (?, mp_dim, embedding_size)
        """
        # dimension of vector
        n_dim = K.ndim(x)
        embedding_size = K.int_shape(x)[-1]
        timesteps = K.int_shape(x)[1]
        if timesteps is None:
            timesteps = K.shape(x)[1]

        # collapse time dimension and batch dimension together
        x = K.reshape(x, (-1, embedding_size))
        # reshape to (?, 1, embedding_size)
        x = K.expand_dims(x, axis=1)
        # reshape weights to (1, mp_dim, embedding_size)
        w = K.expand_dims(w, axis=0)
        # element-wise multiply
        x = x * w
        # reshape to original shape
        if n_dim == 3:
            x = K.reshape(x, K.stack([-1, timesteps, self.mp_dim, embedding_size]))
            x.set_shape([None, None, None, embedding_size])
        elif n_dim == 2:
            x = K.reshape(x, K.stack([-1, self.mp_dim, embedding_size]))
            x.set_shape([None, None, embedding_size])
        return x

    def _full_matching(self, h1, h2, w):
        """Full matching operation.
        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            w: weights of one direction, (mp_dim, embedding_size)
        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h2 forward last step hidden vector, (batch_size, embedding_size)
        h2_last_state = h2[:, -1, :]
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # h2_last_state * weights, (batch_size, mp_dim, embedding_size)
        h2 = self._time_distributed_multiply(h2_last_state, w)
        # reshape to (batch_size, 1, mp_dim, embedding_size)
        h2 = K.expand_dims(h2, axis=1)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, h2)
        return matching

    def _max_pooling_matching(self, h1, h2, w):
        """Max pooling matching operation.
        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            w: weights of one direction, (mp_dim, embedding_size)
        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # h2 * weights, (batch_size, h2_timesteps, mp_dim, embedding_size)
        h2 = self._time_distributed_multiply(h2, w)
        # reshape v1 to (batch_size, h1_timesteps, 1, mp_dim, embedding_size)
        h1 = K.expand_dims(h1, axis=2)
        # reshape v1 to (batch_size, 1, h2_timesteps, mp_dim, embedding_size)
        h2 = K.expand_dims(h2, axis=1)
        # cosine similarity, (batch_size, h1_timesteps, h2_timesteps, mp_dim)
        cos = self._cosine_similarity(h1, h2)
        # (batch_size, h1_timesteps, mp_dim)
        matching = K.max(cos, axis=2)
        return matching

    def _attentive_matching(self, h1, h2, cosine_matrix, w):
        """Attentive matching operation.
        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            cosine_matrix: weights of hidden state h2,
                          (batch_size, h1_timesteps, h2_timesteps)
            w: weights of one direction, (mp_dim, embedding_size)
        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # attentive vector (batch_size, h1_timesteps, embedding_szie)
        attentive_vec = self._mean_attentive_vectors(h2, cosine_matrix)
        # attentive_vec * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        attentive_vec = self._time_distributed_multiply(attentive_vec, w)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, attentive_vec)
        return matching

    def _max_attentive_matching(self, h1, h2, cosine_matrix, w):
        """Max attentive matching operation.
        # Arguments
            h1: (batch_size, h1_timesteps, embedding_size)
            h2: (batch_size, h2_timesteps, embedding_size)
            cosine_matrix: weights of hidden state h2,
                          (batch_size, h1_timesteps, h2_timesteps)
            w: weights of one direction, (mp_dim, embedding_size)
        # Output shape
            (batch_size, h1_timesteps, mp_dim)
        """
        # h1 * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        h1 = self._time_distributed_multiply(h1, w)
        # max attentive vector (batch_size, h1_timesteps, embedding_szie)
        max_attentive_vec = self._max_attentive_vectors(h2, cosine_matrix)
        # max_attentive_vec * weights, (batch_size, h1_timesteps, mp_dim, embedding_size)
        max_attentive_vec = self._time_distributed_multiply(max_attentive_vec, w)
        # matching vector, (batch_size, h1_timesteps, mp_dim)
        matching = self._cosine_similarity(h1, max_attentive_vec)
        return matching


# In[4]:


encode_dim = 32 * 2
def Self_Encoder(w2v):
    class NonMasking(Layer):   
        def __init__(self, **kwargs):   
            self.supports_masking = True  
            super(NonMasking, self).__init__(**kwargs)   

        def build(self, input_shape):   
            input_shape = input_shape   

        def compute_mask(self, input, input_mask=None):   
            # do not pass the mask to the next layers   
            return None   

        def call(self, x, mask=None):   
            return x   

        def get_output_shape_for(self, input_shape):   
            return input_shape 
    sentence_input = Input(shape=([48]))
    embedding = Embedding(input_dim=len(w2v), output_dim=300, mask_zero=True, trainable=False)
    embedding.build((None,)) # if you don't do this, the next step won't work
    embedding.set_weights([np.array(w2v)])
    
    seq = NonMasking()(embedding(sentence_input))
    emb = Bidirectional(GRU(int(encode_dim / 2), dropout=0.3, name='gru_1',                            kernel_regularizer=regularizers.l2(0.005), return_sequences = True))(seq)
    emb = TimeDistributed(Highway(activation='tanh'))(emb)
    #return the encoding for both two layer (soft connect)
    return Model(inputs=sentence_input, outputs=emb)


def Auto_encoder(voc_size, encoder):
    sentence_input = Input(shape=([48]))
    enc_input = GlobalAveragePooling1D()(encoder(sentence_input))
    out = Dense(int(voc_size), activation='softmax')(Dropout(0.5)(enc_input))
    return Model(inputs=sentence_input, outputs=out)


class Encoder_w_highway(object):
    """Word context layer
    """
    def __init__(self, rnn_dim, rnn_unit='gru', input_shape=(0,), dropout=0.0, highway=False, return_sequences=False, dense_dim=0):
        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(Bidirectional(rnn(rnn_dim, dropout=dropout, recurrent_dropout=dropout,
                              return_sequences=return_sequences), input_shape=input_shape))
        if highway:
            if return_sequences:
                self.model.add(TimeDistributed(Highway(activation='tanh')))
            else:
                self.model.add(Highway(activation='tanh'))

        if dense_dim > 0:
            self.model.add(TimeDistributed(Dense(dense_dim, activation='relu')))
            self.model.add(TimeDistributed(Dropout(dropout)))
            self.model.add(TimeDistributed(BatchNormalization()))
    def __call__(self, inputs):
        return self.model(inputs)

def Att():
    emb = Input(shape=([48, encode_dim]))
    att = TimeDistributed(Dense(1, activation='sigmoid'))(emb)
    att = Flatten()(att)
    atp = RepeatVector(encode_dim)(att)
    atp = Permute((2,1))(atp)
    out = Multiply()([atp, emb])
    return Model(inputs=emb, outputs=[out, att])    
    

def Matcher(enc_l, enc_r, latt, ratt):
    inp1 = Input(shape=([48]))
    inp2 = Input(shape=([48]))
    
    emb11, _ = latt(enc_l(inp1))    
    emb21, _ = ratt(enc_r(inp2))

    matching_layer1 = MultiPerspective(10)
 
    matching_1 = matching_layer1([emb11, emb21])
    matching_2 = matching_layer1([emb21, emb11])

    
    agg = Encoder_w_highway(16, highway=True, input_shape=(48, K.int_shape(matching_1)[-1],), return_sequences=False)
    m1 = agg( matching_1 )
    m2 = agg( matching_2 )
    out = concatenate([m1, m2])
    out = Dense(1, activation='sigmoid')(Dropout(0.5)(out))
    return Model(inputs=[inp1, inp2], outputs=out)

es_encoder = Self_Encoder(es2vec)
self_cmper = Comparer()
es_att = Att()
es_matcher = Matcher(es_encoder, es_encoder, self_cmper, es_att)
es_matcher.compile(optimizer='rmsprop',          loss='binary_crossentropy')
es_autoencoder = Auto_encoder(es_cnt, es_encoder)
es_autoencoder.compile(optimizer='rmsprop',
              loss='kullback_leibler_divergence')

# In[ ]:


n_folds = 5

X, y = np.array(zip(es1, es2)), np.array(lab)
batch_size = 320
train_epoc = 50

skf = StratifiedKFold(n_splits=n_folds, random_state=43, shuffle=True)
stack_preds = {i : [] for i in range(n_folds)}
test_preds  = {i : [] for i in range(n_folds)}
n_fold = -1

neg_aug1 = np.concatenate((es1, esu, test1, test2))
neg_aug2 = np.concatenate((es2, esu, test1, test2))
if os.path.exists('./checkpoint/bimpm'):
    shutil.rmtree('./checkpoint/bimpm', ignore_errors=True)
os.makedirs('./checkpoint/bimpm')
for train_index, test_index in skf.split(X, y):
    es_att = Att()
    
    # es_encoder = Self_Encoder(es2vec)
    es_matcher = Matcher(es_encoder, es_encoder, es_att, es_att)

    es_encoder.load_weights('./es_pretrain_encoder.h5')
    es_matcher.compile(optimizer='rmsprop',              loss='binary_crossentropy')
    
    
    n_fold = n_fold + 1
    os.makedirs('./checkpoint/mp2/' + str(n_fold))
    esX_train, esX_valid = X[train_index], X[test_index]
    esy_train, esy_valid = y[train_index], y[test_index]
    augX_train, augX_valid, augy_train, augy_valid = train_test_split(zip(es_aug1, es_aug2),                     [1 for i in range(len(es_aug1))], test_size=0.1)

    fesX_train = np.concatenate((esX_train, augX_train))
    fesy_train = np.concatenate((esy_train, augy_train))
    
    aug_iter = int((np.sum(fesy_train) - 0.3 * len(fesy_train)) / 0.3)
    
    print(fesX_train.shape, esX_valid.shape)
    aw = 1
    bw = 1
    cw = 0.5
    dw = 0.5

    for epoch in range(train_epoc): 
        print('start train for epoch ' + str(epoch))
        #start train
        print('spanish match')
        aug_x = []
        aug_y = []
        for i, j in np.random.choice(range(len(neg_aug1)), 2 * aug_iter).reshape(-1, 2):
            if i != j:
                aug_x += [[neg_aug1[i], neg_aug2[j]]]
                aug_y += [0]
        xt  = np.concatenate((fesX_train, aug_x))
        yt  = np.concatenate((fesy_train, aug_y))
        
        print(xt.shape, np.sum(yt) / float(len(yt)))

        weight = np.array([aw if yi == 1 else bw for yi in esy_train] +                           [dw for yi in augy_train] + [cw for yi in aug_y])
        
        es_matcher.fit([xt[:,0,:], xt[:,1,:]], yt, shuffle=True,
                       validation_data=([esX_valid[:, 0, :], esX_valid[:,1,:]], esy_valid),
                       batch_size = batch_size, sample_weight = weight)      

        #end train, start evaluate
        valid_pred = es_matcher.predict([esX_valid[:,0,:],esX_valid[:, 1, :]],                                   batch_size=batch_size, verbose=True).reshape(-1)


        aug_x = []
        aug_y = []
        for xi in esX_valid:
            aug_x += [[xi[np.random.randint(2)], es[np.random.randint(len(es))]]]
            aug_y += [0]   
            
        res2 = es_matcher.predict([np.array(aug_x)[:,0,:], np.array(aug_x)[:,1,:]],                                   verbose=True, batch_size=batch_size)       
        res3 = es_matcher.predict([np.array(augX_valid)[:,0,:], np.array(augX_valid)[:,1,:]],                                   verbose=True, batch_size=batch_size)       

        a = []
        b = []
        c = []
        d = []
        for i, j in zip(esy_valid, valid_pred):
            if i == 1:
                a += [-np.log(j)]
            else: 
                b += [-np.log(1-j)]
        for j in res2:
            c += [-np.log(1-j)]

        for j in res3:
            d += [-np.log(j)]

            
        om = np.average(a+b) 
        
        
        sb.distplot(valid_pred)
        plt.title('valid---1: %3f 0: %3f aug0: %3f aug1: %3f all: %3f' % (np.average(a), np.average(b),                                     np.average(c), np.average(d), om))
        plt.show()
        
        
        aai, bbi, cci, ddi = (np.average(a), np.average(b), np.average(c), np.average(d)) / om
        aw, bw, cw, dw = (aw + 0.2 * aai) / 1.2, (bw + 0.2 * bbi) / 1.2, (cw + 0.2 * cci) / 1.2, (dw + 0.2 * ddi) / 1.2
        print(aai, bbi, cci, ddi)
        print(aw, bw, cw, dw)
        
        stack_preds[n_fold] += [valid_pred]
        
        test_pred = es_matcher.predict([test1, test2], batch_size=batch_size, verbose=True).reshape(-1)
        test_preds[n_fold] += [test_pred]
        
        sb.distplot(test_pred)
        plt.title('test: ' + str((np.sum(test_pred > 0.5) / float(len(test_pred)))) + '     ' +                  str(-np.average(log_loss(gt[:, 1] > 0.5, test_pred[np.array(gt[:, 0], dtype=np.int)]))))
        plt.show()
        es_matcher.save_weights("./checkpoint/mp2/%d/%s.h5" % (n_fold, epoch))


# In[ ]:


n_fold = -1
os.makedirs('./checkpoint/bimpm/result')
for train_index, test_index in skf.split(np.array(zip(es1, es2)), np.array(lab)):
    esy_valid = np.array(lab)[test_index]
    n_fold += 1
    stack_p = stack_preds[n_fold]
    test_p  = test_preds[n_fold]
    f = []
    for pred in stack_p:
        f += [-np.average(log_loss(esy_valid, pred))]
    x = []
    y = []
    for s in (np.argsort(f))[:5]:
        pred = stack_p[s]
        x += [pred]
        sb.distplot(pred)
        plt.title(str(f[s]))
        plt.show()
        
        pred = test_p[s]
        y += [pred]
        sb.distplot(pred)
        plt.title(str(np.sum(pred > 0.5) / float(len(pred))) + ' ' + str(-np.average(log_loss(gt[:, 1] > 0.5,  pred[np.array(gt[:, 0], dtype=np.int)]))))
        plt.show()
        print('----------')
    print('ensemble')
    x = np.average(x, axis=0)
    y = np.average(y, axis=0)
    sb.distplot(x)
    plt.title(-np.average(log_loss(esy_valid, x)))
    plt.show()
    sb.distplot(y)
    plt.title(str(np.sum(y > 0.5) / float(len(y))) + ' ' + str(-np.average(log_loss(gt[:, 1] > 0.5,  y[np.array(gt[:, 0], dtype=np.int)]))))
    plt.show()    
    pk.dump(x, open('./checkpoint/bimpm/result/valid_' + str(n_fold), 'w'))
    pk.dump(y, open('./checkpoint/bimpm/result/test_' + str(n_fold), 'w'))


# In[ ]:


f = []
x = []
y = []
for pred in valid_preds:
    a = []
    b = []
    c = []
    d = []
    for i, j in zip(esy_test, pred):
        if i == 1:
            a += [-np.log(j)]
            c += [j]
        else:
            b += [-np.log(1-j)]
            d += [1-j]
    f += [np.average(a + b)]
for s in (np.argsort(f)[:10]):
    pred = valid_preds[s]
    x += [pred]
    a = []
    b = []
    c = []
    d = []
    for i, j in zip(esy_test, pred):
        if i == 1:
            a += [-np.log(j)]
            c += [j]
        else:
            b += [-np.log(1-j)]
            d += [1-j]
    print(str(s) + ' ' + str(np.average(a)) + '  ' + str(np.average(b)) + '  ' +  str(np.average(a+b)))
    pred = preds[s].reshape(-1)
    y += [pred]
    sb.distplot(pred)
    plt.show()
    print(np.sum(pred > 0.5) / float(len(pred)))
    print(-np.average(log_loss(gt[:, 1] > 0.5,  pred[np.array(gt[:, 0], dtype=np.int)])))
    
x = np.array(x).T
y = np.array(y).T


# In[ ]:


pk.dump(x, open('../archive/valid_pred_bimpm_att', 'w'))
pk.dump(y, open('../archive/test_pred_bimpm_att', 'w'))


# In[ ]:


feature_extraction = Model(inputs=es_matcher.input, outputs=es_matcher.layers[-3].output)
feat = feature_extraction.predict([es1, es2])
pk.dump(feat, open('../archive/feat_bimpm_att', 'w'))
feat = feature_extraction.predict([test1, test2])
pk.dump(feat, open('../archive/test_feat_bimpm_att', 'w'))


# In[ ]:


att_dict = {i: [] for i in es_sts.keys()}
for i in range(len(es) / 1000):
    print(i, len(es) / 1000)
    ss = es[i * 1000: (i+1) * 1000]
    enc = es_encoder.predict(np.array(ss))
    att = es_att.predict(enc)
    for s, a in zip(ss, att[1]):
        for w, ai in zip(s, a):
            if w == 0:
                break
            att_dict[w] += [ai]
stat = []
for i in att_dict:
    stat += [(i, np.average(att_dict[i]), np.var(att_dict[i]))]
stat.sort(key=lambda x: -x[1])
for i in stat:
    if i[0] != 0:
        print(id2es[i[0]])
        print(i[1], i[2])


# In[ ]:


es2id = pk.load(open('../data/es2id.dat'))
id2es = {es2id[i] : i for i in es2id}
enc = es_encoder.predict(es1[:100])
at1 = es_att.predict(enc)[1]

enc = es_encoder.predict(es2[:100])
at2 = es_att.predict(enc)[1]

res = es_matcher.predict([es1[:100], es2[:100]])
res = log_loss(out[:100], res.reshape(-1))
for s1, s2, att1, att2, o, l in zip(es1, es2, at1, at2, res, out):
    if l == 1:
        print(o, l)
        print(' '.join([id2es[s] for s in s1 if s != 0]))
        print(' '.join([id2es[s] for s in s2 if s != 0]))
        for s, a in zip(s1, att1):
            if s == 0:
                break
            print(id2es[s] + ' ' + str(a))
        print('---------')
        for s, a in zip(s2, att2):
            if s == 0:
                break
            print(id2es[s] + ' ' + str(a))


# In[ ]:


dis = (res1 * res2).reshape(-1)
for i in np.argsort(-dis)[:5]:
    print(translate(epu[i]))


# In[ ]:


es_matcher.save('../archive/bimpm.h5')


# In[ ]:


def translate(l):
    return ' '.join([id2es[i] for i in l if i != 0])


# In[ ]:


s1, s2 = esX_test[:,0,:], esX_test[:,1,:]


# In[ ]:


res = es_matcher.predict([s1,s2], batch_size=batch_size).reshape(-1)


# In[ ]:


gaps = []
geat = []
for tl, pr, sa, sb in zip(esy_test, res, s1, s2):
#     print(translate(sa))
#     print(translate(sb))
    res1 = es_matcher.predict([epu, np.array([sa for j in range(len(epu))])], batch_size=640, verbose=False)
    res2 = es_matcher.predict([epu, np.array([sb for j in range(len(epu))])], batch_size=640, verbose=False)
    dis = (res1 * res2).reshape(-1)
    f = [pr, np.max(dis), np.average(dis[np.argsort(-dis)[:3]]), np.average(dis[np.argsort(-dis)[:5]]),          np.average(dis[np.argsort(-dis)[:10]]), np.average(dis[np.argsort(-dis)[:20]])]
    geat += [f]
    print(f, tl)
#     print('------------')
#     for i in np.argsort(-dis)[:5]:
#         print(res1[i][0], res2[i][0], dis[i])
#         print(translate(epu[i]))
#     print('====================')


# In[ ]:


sgd = optimizers.sgd(lr=0.01)
inp = Input(shape=([6]))
# out = Dense(6, activation='tanh')(inp)
out = Dense(1, activation='sigmoid')(inp)
merge_model = Model(inputs=inp, outputs=out)
merge_model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


merge_model.fit(np.array(geat), esy_test, epochs=500, verbose=2, batch_size=512)


# In[ ]:


merge_model.layers[1].get_weights()

