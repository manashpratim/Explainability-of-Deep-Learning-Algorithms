import numpy as np
import tensorflow as tf

from typing import Tuple

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import \
    Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Input, Model

from hw4_mnist import MNISTModel


class MNISTModelRegular(MNISTModel):
    """A version of an MNIST model to instrument the pre-activation output of the
       last intermediate layer (the one before the one that produces logits,
       the softmax layer has no trainable parameters) and adds L2
       regularization to the learnable parameters of this layer. We refer to the
       output of this layer as "features".
    """

    def __init__(self, lam=0.1, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        self.lam = lam

    def build(self):
        # Running this will reset the model's parameters
        layers = []


        layers.append(Conv2D(
            filters=16,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation='relu'
        ))

        layers.append(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        layers.append(Conv2D(
            filters=32,
            kernel_size=(4, 4),
            padding='same',
            activation='relu'
        ))

        layers.append(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        layers.append(Flatten())

        layers.append(Dense(units=64,name="features")
        )
        layers.append(Activation('relu'))

        # layers[-2]
        layers.append(Dense(
            units=self.num_classes,
            name="logits",
            kernel_regularizer= regularizers.l2(self.lam),bias_regularizer = tf.keras.regularizers.l2(self.lam) 
        ))

        # layers[-1]
        layers.append(Activation('softmax'))



        self.layers = layers

        self._define_ops()
        self.X = Input(shape=(28, 28, 1))
        self.Ytrue = Input(shape=(self.num_classes), dtype=tf.int32)

        self.tensors = self.forward(self.X, self.Ytrue)

        self.model = Model(self.X, self.tensors['probits'])

        # models can be symbolically composed
        self.features = Model(self.X, self.tensors['features'])
        self.logits = Model(self.X, self.tensors['logits'])
        self.probits = Model(self.X, self.tensors['probits'])
        self.preds = Model(self.X, self.tensors['preds'])

        # functions evaluate to concrete outputs
        self.f_features = K.function(self.X, self.tensors['features'])
        self.f_logits = K.function(self.X, self.tensors['logits'])
        self.f_probits = K.function(self.X, self.tensors['probits'])
        self.f_preds = K.function(self.X, self.tensors['preds'])



    def forward(self, X, Ytrue=None):
        _features: tf.Tensor = None # new tensor to build
        _logits: tf.Tensor = None
        _probits: tf.Tensor = None
        _preds: tf.Tensor = None
        _loss: tf.Tensor = None

        # >>> Your code here <<<
        c = X
        parts = []
        for l in self.layers:
            c = l(c)
            parts.append(c)
        _features = parts[-4]
        _logits = parts[-2]
        _probits = parts[-1]

        _preds = tf.argmax(_probits, axis=1)

        if Ytrue is not None:
            # Same as the loss specified in train below.
            _loss = K.mean(K.sparse_categorical_crossentropy(
                self.Ytrue,
                _probits
            ))


        return {
            'features': _features,
            'logits': _logits,
            'probits': _probits,
            'preds': _preds,
            'loss': _loss,
        }

    def load(self, batch_size=16, filename=None):
        if filename is None:
            filename = f"model.regular{self.lam}.MNIST.h5"

        super().load(batch_size=batch_size, filename=filename)

    def save(self, filename=None):
        if filename is None:
            filename = f"model.regular{self.lam}.MNIST.h5"

        super().save(filename=filename)


class Representer(object):
    def __init__(
        self,
        model: MNISTModelRegular,
        X: np.ndarray,
        Ytrue: np.ndarray
    ) -> None:
        """
        X: np.ndarray [N, 28, 28, 1] training points
        Y: np.ndarray [N] ground truth labels
        """

        assert "features" in model.tensors, \
            "Model needs to provide features tensor."
        assert "loss" in model.tensors, \
            "Model needs to provide loss tensor."
        assert "logits" in model.tensors, \
            "Model needs to provide logits tensor."

        self.model = model
        self.lam = model.lam
        self.X = X
        self.Ytrue = Ytrue

        self._define_ops()

    def _define_ops(self):


        self.features = self.model.features.predict(self.X)
        self.num_classes = len(np.unique(self.Ytrue))
        self.N = self.X.shape[0]
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.y = tf.placeholder(tf.int64, shape=[None, 1])
        self.logits = tf.placeholder(tf.float32, shape=[None, 10])
        self.labels = tf.one_hot(self.y, depth=self.num_classes)
        self.loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits = self.logits, dim=-1))
        self.gradient = tf.gradients(ys=self.loss, xs = self.logits)[0]




    def similarity(self, Xexplain: np.ndarray) -> np.ndarray:
        """For each input instance, compute the similarity between it and every one of
        the training instances. This is the f_i f_t^T term in the paper.

        inputs:
            Xexplain: np.ndarray [M, 28, 28, 1] -- images to explain

        return
            np.ndarray [M, N]

        """


        similar= np.zeros((Xexplain.shape[0],self.N))

        for i in range(Xexplain.shape[0]):

            features_target = self.model.features.predict(np.expand_dims(Xexplain[i],0))
            similar[i] = np.dot(features_target,self.features.T)

        return similar

        

    def coeffs(self) -> np.ndarray:
        """For each training instance, compute its representer value coefficient. This
        is the alpha term in the paper.

        inputs:
            none

        return
            np.ndarray [N, 10]

        """
 

        # >>> Your code here <<<
        gradients_logits =np.array([self.sess.run(self.gradient,feed_dict= {self.x:self.X[i].reshape(1,28,28,1),
                                   self.y:self.Ytrue[i].reshape(-1,1),
                                   self.logits:self.model.logits.predict(self.X[i].reshape(1,28,28,1)),
                            }
        )for i in range(self.X.shape[0])])

        coeff = -gradients_logits/(2*self.lam*self.X.shape[0])
        coeff = np.squeeze(coeff, axis=1)

        return coeff


    def values(self, coeffs, sims):
        """Given the training instance coefficients and train/test feature
        similarities, compute the representer point values. This is the k term
        from the paper.

        inputs:
            coeffs: np.ndarray [N, 10]
            sims: np.ndarray [M, N]

        return
            np.ndarray [M, N, 10]

        """

        value = np.zeros((sims.shape[0],sims.shape[1],coeffs.shape[-1]))
        for i in range(sims.shape[0]):
            value[i]  = sims[i].reshape(-1,1)*coeffs

        return value

    def coeffs_and_values(
        self,
        Xexplain: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For each input instance, compute the representer point coefficients of the
        training data(self.X, self.Y)

        inputs:
            Xexplain: np.ndarray [M, 28, 28, 1] -- images to explain
            target: target class being explained

        returns:
            coefficients: np.ndarray of size [N, 10]
            values: np.ndarray of size [M, N, 10]
              N is size of |self.X|

        """

        coeffs = self.coeffs()
        sims = self.similarity(Xexplain)

        return coeffs, self.values(coeffs, sims)
