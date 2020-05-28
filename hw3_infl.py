import numpy as np
import tensorflow as tf
from vgg16 import vgg16
from imagenet_classes import class_names
import scipy
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.backend import resize_images
from hw3_utils import *
import matplotlib.pyplot as plt
import matplotlib as mpl


class InflDirected(object):
    """Implementaion of Influence Directed Explanation

        https://arxiv.org/abs/1802.03788
    """
    def __init__(self, sess,
                 input_tensor, internal_layer,
                 output_tensor, qoi):
        """__Constructor__

        Arguments:
            sess {tf.Session} -- Tensorflow session
            input_tensor {tf.Tensor} -- Symbolic tensor of (batch of) inputs
            internal_layer {tf.Tensor} -- Symbolic tensor of internal layer
              output
            output_tensor {tf.Tensor} -- Symbolic tensor of (batch of)
              pre-softmax outputs
            qoi {tf.Tensor -> tf.Tensor} -- The quantity of interest, a
              function that takes in a single instance output and produces
              the quantity of interest tensor for that output.

            Example usage of qoi:
              InflDirected(sess, input_batch, layer,
                           output_batch, qoi=lambda out: out[target_class])
            # for some target class target_class

        """
        #qoi = = lambda out: out[tf.keras.backend.argmax(out)]
        self.sess = sess
        self.input_tensor = input_tensor
        self.internal_layer = internal_layer
        self.output_tensor = output_tensor

        self.qoi = tf.map_fn(qoi, self.output_tensor)

        self._define_ops()

    def _define_ops(self):


        self.expert = tf.placeholder(tf.int32)
        self.grad = tf.gradients(xs=self.internal_layer, ys=self.qoi)[0]
        self.ena = tf.gradients(xs=self.input_tensor, ys=self.internal_layer[:,self.expert])[0]

    def dis_influence(self, X, batch_size=16):
        """Compute the distribution of influence and return the expert neuron

        Arguments:
            X {np.ndarray} -- Input dataset

        Keyword Arguments:
            batch_size {int} -- Batch Size (default: {16})

        Returns:
            int -- The expert neuron in the internal layer.
        """
        expert_id = None


        d = []
        for b in range(0, len(X), batch_size):

            d = d + [self.sess.run(self.grad, feed_dict={self.input_tensor: X[b:b + batch_size]})]

        doi = np.array([j for i in range(len(d)) for j in d[i]])

        expert_id = np.argmax(np.average(doi,axis=0))

        return expert_id

    def expert_attribution(self,
                           expert_id,
                           X,
                           batch_size=16,
                           multiply_with_input=True):
        """__call__ forward computation to generate the saliency map

        Arguments:
            expert_id {int} -- The nueron index of expert
            X {np.ndarray} -- Input dataset

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {16})
            multiply_with_input {bool} -- If True, return grad x input,
                else return grad (default: {True})

        Returns np.ndarray of the same shape as X.
        """
        attr_map = np.zeros_like(X)

        gradients = np.zeros(X.shape)
        for b in range(0, len(X), batch_size):
            gradients[b:b + batch_size] = self.sess.run(self.ena, feed_dict={self.input_tensor: X[b:b + batch_size],self.expert:expert_id}) 

        attr_map = X*gradients
        
        if multiply_with_input:
            return attr_map
        else:
            return gradients

            

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'imagenet', sess)
    dataset = load_ImageNet()
    images = ['laska.png', 'camel.png']

    #imgs = np.array([img_to_array(load_img(i, target_size=(224, 224))) for i in images])
    input_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    output_tensor = vgg.fc3l
    internal_layer = vgg.fc1
    imgs = np.array([dataset[0][dataset[1]==511][2],dataset[0][dataset[1]==386][7]])
    target = [511,386]

    for tar in  range(len(target)):
        print(target[tar])
        attr_fn = InflDirected(sess, input_tensor, internal_layer,output_tensor, qoi=lambda out: out[target[tar]])
        neuron=attr_fn.dis_influence(dataset[0][dataset[1] == target[tar]], batch_size=16)
        #neuron=attr_fn.dis_influence(dataset[0], batch_size=16)
        #print(neuron)
        attr = attr_fn.expert_attribution(neuron,np.expand_dims(imgs[tar],0))
        k = point_cloud(attr, threshold=0)
        f = np.zeros((attr.shape))
        f[:,:,:,0] = k
        f[:,:,:,1] = k
        f[:,:,:,2] = k

        x = binary_mask(np.expand_dims(imgs[tar],0), f,norm=True,threshold=0.2,blur=2,background=0.1)

        fig, ax = plt.subplots(1,3, figsize=(20, 3))
        fig.suptitle('Influence Directed Explanations')

        ax[0].imshow(imgs[tar]/255)
   
        ax[0].set_title('Input')

        ax[1].imshow(k[0])
        
        ax[1].set_title('Expert at fc_1')

        ax[2].imshow(x[0])
        
        ax[2].set_title('Visualizations')

       
        plt.savefig('part2e'+str(tar)+'.png')
        plt.show()


    #Part D

    attr_fn = InflDirected(sess, input_tensor, internal_layer,output_tensor, qoi=lambda out: out[511] - out[386])
    #neuron=attr_fn.dis_influence(dataset[0][dataset[1] == 511], batch_size=16)
    neuron=attr_fn.dis_influence(dataset[0], batch_size=16)
    print(neuron)
   
    attr = attr_fn.expert_attribution(neuron,np.expand_dims(dataset[0][dataset[1]==511][2],0))
    k = point_cloud(attr, threshold=0)
    f = np.zeros((attr.shape))
    f[:,:,:,0] = k
    f[:,:,:,1] = k
    f[:,:,:,2] = k

    x = binary_mask(np.expand_dims(dataset[0][dataset[1]==511][2],0), f,norm=True,threshold=0.2,blur=2,background=0.1)

    fig, ax = plt.subplots(1,3, figsize=(20, 3))
    fig.suptitle('Influence Directed Explanations')

    ax[0].imshow(dataset[0][dataset[1]==511][2]/255)

    ax[0].set_title('Input')

    ax[1].imshow(k[0])
    
    ax[1].set_title('Expert at fc_1')

    ax[2].imshow(x[0])
    
    ax[2].set_title('Visualizations')

   
    plt.savefig('part2Da1'+str(0)+'.png')
    plt.show()


