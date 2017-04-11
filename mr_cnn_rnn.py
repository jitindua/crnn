'''

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python

'''

from __future__ import print_function
import numpy as np
np.random.seed(3435)  # for reproducibility, should be first


from keras import initializers, regularizers, activations, constraints
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential#, Graph
from keras.layers import Dropout, Activation, Flatten, \
    Embedding, Conv1D, MaxPooling1D, \
    Input, Dense, TimeDistributed, Conv2D, MaxPooling2D, Concatenate, Reshape, Bidirectional
from keras.regularizers import l2
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.constraints import maxnorm
from keras.datasets import imdb
from keras import callbacks
from keras.utils import generic_utils
from keras.models import Model
from keras.optimizers import Adadelta, SGD
import time


word_model = 1
char_model = 1
char_vec_size = 15
batch_size = 50
nb_filter = 200
filter_length = 4
hidden_dims = nb_filter * 2
nb_epoch = 200
RNN = GRU
rnn_output_size = 100
folds = 10
checkpoint_dir = 'cv'
savefile = 'char-large'

print('Loading data...')

import mr_data
if word_model:
  X_train_w, y_train_w, X_test_w, y_test_w, W, W2 = mr_data.load_data(fold=0, word_model = 1, char_model=0)
  max_features = len(W)
  embedding_dims = len(W[0])
  print('word_vocab_size', max_features)
  print('max_sent_l', embedding_dims)
  print('X_train word shape:', X_train_w.shape)
  print('X_test word shape:', X_test_w.shape)
if char_model:
  X_train, y_train, X_test, y_test, char_vocab_size, max_word_l = mr_data.load_data(fold=0, word_model = 0, char_model=1)
  print('char_vocab_size', char_vocab_size)
  print('max_word_l', max_word_l)
  print('X_train chars shape:', X_train.shape)
  print('X_test chars shape:', X_test.shape)

if (word_model==1) and (char_model==0):
    X_train = X_train_w
    y_train = y_train_w
    X_test = X_test_w
    y_test = y_test_w
elif (word_model and char_model):
    X_train = {'word':X_train_w, 'chars':X_train}
    y_train = {'output':y_train}
    X_test = {'word':X_test_w, 'chars':X_test}
    y_test = {'output':y_test}

# char:
# X_train shape: (9595, 64, 20)
# X_test shape: (1067, 64, 20)
# word: 
# X_train shape: (9595, 64)
# X_test shape: (1067, 64)





class Highway(Layer):
    """Densely connected highway network.
    Highway layers are a natural extension of LSTMs to feedforward networks.
    # Arguments
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/abs/1505.00387v2)
    """

    def __init__(self,
                 init='glorot_uniform',
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):

        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    shape=(None, input_dim))

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W_carry = self.add_weight((input_dim, input_dim),
                                       initializer=self.init,
                                       name='W_carry')
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_carry = self.add_weight((input_dim,),
                                           initializer='one',
                                           name='b_carry')
        else:
            self.b_carry = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x):
        y = K.dot(x, self.W_carry)
        if self.bias:
            y += self.b_carry
        transform_weight = activations.sigmoid(y)
        y = K.dot(x, self.W)
        if self.bias:
            y += self.b
        act = self.activation(y)
        act *= transform_weight
        output = act + (1 - transform_weight) * x
        return output

    def get_config(self):
        config = {'init': initializers.serialize(self.init),
                  'activation': activations.serialize(self.activation),
                  'W_regularizer': regularizers.serialize(self.W_regularizer),
                  'b_regularizer': regularizers.serialize(self.b_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'W_constraint': constraints.serialize(self.W_constraint),
                  'b_constraint': constraints.serialize(self.b_constraint),
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class sSGD(SGD):
    def __init__(self, scale=1., **kwargs):
        super(sSGD, self).__init__(**kwargs)
        self.scale = scale;
    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if self.scale != 1.:
            grads = [g*K.variable(self.scale) for g in grads]
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [K.switch(norm >= self.clipnorm, g * self.clipnorm / norm, g) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads
    
class sModel(Model):
    def fit_generator(self, generator, steps_per_epoch, epochs, validation_data, validation_steps, opt=None):
        learning_rate_decay = 0.5
        save_every = 5
        decay_when = 1
        val_losses = []
        lr = K.get_value(self.optimizer.lr)
        for epoch in range(epochs):
            super(sModel, self).fit_generator(generator, steps_per_epoch, epochs=epoch+1, verbose=1, initial_epoch=epoch)
            val_loss = exp(self.evaluate_generator(validation_data, validation_steps))
            val_losses.append(val_loss)
            print ('Epoch {}/{}. Validation loss: {}'.format(epoch + 1, epochs, val_loss))
            if len(val_losses) > 2 and (val_losses[-2] - val_losses[-1]) < decay_when:
                lr *= learning_rate_decay
                K.set_value(self.optimizer.lr, lr)
            if epoch == epochs-1 or epoch % save_every == 0:
                savefile = '%s/lm_%s_epoch%d_%.2f.h5' % (checkpoint_dir, savefile, epoch + 1, val_loss)
                self.save_weights(savefile)
    @property
    def state_updates_value(self):
        return [K.get_value(a[0]) for a in self.state_updates]

    def set_states_value(self, states):
        return [K.set_value(a[0], state) for a, state in zip(self.state_updates, states)]

    def save(self, name):
        json_string = self.to_json()
        with open(name, 'wt') as f:
            f.write(json_string)



if word_model and char_model:
    # sent len + pad
    maxlen = X_train['word'].shape[1]
else:
    maxlen = X_train.shape[1]


print('Train...')
accs = []
first_run = True
for i in xrange(folds):

    
    if word_model:
          X_train_w, y_train_w, X_test_w, y_test_w, W, W2 = mr_data.load_data(fold=i, word_model = 1, char_model=0)
    if char_model:
          X_train, y_train, X_test, y_test, char_vocab_size, max_word_l = mr_data.load_data(fold=i, word_model = 0, char_model=1)
    rand_idx = np.random.permutation(range(len(X_train)))

    if word_model:
        X_train_w = X_train_w[rand_idx]
        y_train_w = y_train_w[rand_idx]
    if char_model:
        X_train = X_train[rand_idx]
        y_train = y_train[rand_idx]
    if (word_model==1) and (char_model==0):
        X_train = X_train_w
        y_train = y_train_w
        X_test = X_test_w
        y_test = y_test_w
    elif (word_model and char_model):
        X_train = {'word':X_train_w, 'chars':X_train}
        y_train = {'output':y_train}
        X_test = {'word':X_test_w, 'chars':X_test}
        y_test = {'output':y_test}

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    def CNN(seq_length, length, input_size, feature_maps, kernels, x):
        
        concat_input = []
        for feature_map, kernel in zip(feature_maps, kernels):
            reduced_l = length - kernel + 1
            conv = Conv2D(feature_map, (1, kernel), activation='tanh', data_format="channels_last")(x)
            maxp = MaxPooling2D((1, reduced_l), data_format="channels_last")(conv)
            concat_input.append(maxp)

        x = Concatenate()(concat_input)
        x = Reshape((seq_length, sum(feature_maps)))(x)
        return x

    def charCNN():

        #feature_maps = [50,100,150,200,200,200,200]
        #kernels = [1,2,3,4,5,6,7]
        feature_maps = [25,50,75,100,125,125]
        kernels = [1,2,3,4,5,6]
        
        # chars = Input(batch_shape=(opt.batch_size, opt.seq_length, opt.max_word_l), dtype='int32', name='chars')
        # ? max_len x max_word_l x ?char_vocab_size?
        # chars = Input(shape=(max_len, max_word_l), dtype='int32', name='chars') or
        chars = Input(shape=(maxlen, max_word_l), dtype='int32', name='chars')
        # ? input_length should be max_len x max_word_l. maybe auto determined by TimeDistributed
        chars_embedding = TimeDistributed(Embedding(char_vocab_size, char_vec_size, name='chars_embedding'))(chars)
        #chars_embedding = Embedding(char_vocab_size, char_vec_size, input_length = (maxlen, max_word_l), name='chars_embedding')(chars)
        #chars_embedding = Embedding(maxlen, char_vocab_size, char_vec_size, name='chars_embedding')(chars)
        cnn = CNN(maxlen, max_word_l, char_vec_size, feature_maps, kernels, chars_embedding)

        x = cnn
        inputs = chars

        batch_norm = 0
        if batch_norm:
            x = BatchNormalization()(x)

        highway_layers = 1
        for l in range(highway_layers):
            x = TimeDistributed(Highway(activation='relu'))(x)

        return inputs, x

        


    def build_model():
        print('Build model...%d of %d' % (i + 1, folds))

        if word_model:
            main_input_w = Input(shape=(maxlen, ), dtype='int32', name='word')
            # embedding_dims = 300 (w2v) -> output
            # max_features = vocab
            # max_len: sentece length + padding = 64
            # takes int in range [0, max_features)
            embedding_w  = Embedding(max_features, embedding_dims,
                          weights=[np.matrix(W)], input_length=maxlen,
                          name='embedding')(main_input_w)      
        if char_model:
            # Input would be list of characters instead of index for a word
            # max_features is char representation dimention 26/...
            main_input, embedding = charCNN()

        if word_model and char_model==0:
            main_input = main_input_w
            embedding = embedding_w
        if word_model and char_model:
            embedding = Concatenate()([embedding_w, embedding])
            main_input = [main_input_w, main_input]


        embedding = Dropout(0.50)(embedding)

        cnn_layer = 1
        if cnn_layer:
          conv4 = Conv1D(filters=nb_filter,
                                kernel_size=4,
                                padding='valid',
                                activation='relu',
                                strides=1,
                                name='conv4')(embedding)
          maxConv4 = MaxPooling1D(pool_size=2,
                                   name='maxConv4')(conv4)

          conv5 = Conv1D(filters=nb_filter,
                                kernel_size=5,
                                padding='valid',
                                activation='relu',
                                strides=1,
                                name='conv5')(embedding)
          maxConv5 = MaxPooling1D(pool_size=2,
                                  name='maxConv5')(conv5)

          x = Concatenate()([maxConv4, maxConv5])

          x = Dropout(0.15)(x)
        else:
          x = embedding

        x = RNN(rnn_output_size)(x)
        #x = Bidirectional(RNN(rnn_output_size))(x)
        x = Dropout(0.5)(x)
        #x = LSTM(rnn_output_size)(x)
        #x = Dropout(0.5)(x)
        x = Dense(hidden_dims, activation='relu', kernel_initializer='he_normal',
                  kernel_constraint = maxnorm(3), bias_constraint=maxnorm(3),
                  name='mlp')(x)

        x = Dropout(0.10, name='drop')(x)

        output = Dense(1, kernel_initializer='he_normal',
                       activation='sigmoid', name='output')(x)

        
        #if word_model:
        if 1:
          print ('###1')
          optimizer = Adadelta(lr=0.95, epsilon=1e-06)
          model = Model(inputs=main_input, outputs=output)
          loss = {'output':'binary_crossentropy'}
        else:
          print ('###2')
          model = sModel(inputs=main_input, outputs=output)
          optimizer = sSGD(lr=1.0, clipnorm=5.0, scale = float(maxlen))
          loss = {'binary_crossentropy'}

        model.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=["accuracy"])
        return model

    model = build_model()
    if first_run:
        first_run = False
        print(model.summary())


    '''if not word_model:
      if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
      pickle.dump(opt, open('{}/{}.pkl'.format(checkpoint_dir, savefile), "wb"))
      model.save('{}/{}.json'.format(checkpoint_dir, savefile))'''

    best_val_acc = 0
    best_test_acc = 0
    for j in xrange(nb_epoch):
        a = time.time()
        his = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        validation_split=0.1,
                        shuffle=True,
                        epochs=1, verbose=0)
        print('Fold %d/%d Epoch %d/%d\t%s' % (i + 1,
                                          folds, j + 1, nb_epoch, str(his.history)))
        if his.history['val_acc'][0] >= best_val_acc:
            score, acc = model.evaluate(X_test, y_test,
                                        batch_size=batch_size,
                                        verbose=2)
            best_val_acc = his.history['val_acc'][0]
            best_test_acc = acc
            print('Got best epoch  best val acc is %f test acc is %f' %
                  (best_val_acc, best_test_acc))
            if len(accs) > 0:
                print('Current avg test acc:', str(np.mean(accs)))
        b = time.time()
        cost = b - a
        left = (nb_epoch - j - 1) + nb_epoch * (folds - i - 1)
        print('One round cost %ds, %d round %ds %dmin left' % (cost, left,
                                                               cost * left,
                                                               cost * left / 60.0))
    accs.append(best_test_acc)
    print('Avg test acc:', str(np.mean(accs)))
