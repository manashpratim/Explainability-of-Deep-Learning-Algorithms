import tensorflow as tf
import numpy as np
from vgg16 import vgg16
from imagenet_classes import class_names
import scipy
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.backend import resize_images
from hw3_utils import *
import matplotlib.pyplot as plt
import matplotlib as mpl

class SaliencyMap(object):
    """Implementaion of Saliency Map with Vanilla Gradient

        https://arxiv.org/pdf/1312.6034.pdf

        Example Usage:
        >>> attr_fn = SaliencyMap(sess, input_tensor, output_tensor, 0)
        >>> saliency_map = attr_fn(X) <--- equivalent to attr_fn.__call__(X)
    """
    def __init__(self, sess, input_tensor, output_tensor, target_class):
        """__Constructor__

        Arguments:
            sess {tf.Session} -- Tensorflow session
            input_tensor {tf.Tensor} -- Symbolic tensor of input node
            output_tensor {tf.Tensor} -- Symbolic tensor of pre-softmax output
            target_class {int} -- The class of interest to run explanation.
        """
        self.sess = sess
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.target_class = target_class

        self._define_ops()

    def _define_ops(self):
        """Add whatever operations you feel necessary into the computational
        graph.
        """

        self.y = self.output_tensor[:,self.target_class]
        self.grad = tf.gradients(xs=self.input_tensor, ys=self.y)[0]


    def __call__(self, X, batch_size=16, multiply_with_input=True):
        """__call__ forward computation to generate the saliency map

        Arguments:
            X {np.ndarray} -- Input dataset

        Keyword Arguments:
            batch_size {int} -- Batch size (default: {16})
            multiply_with_input {bool} -- If True, return grad x input,
                else return grad (default: {True})

        Returns np.ndarray of the same shape as X.
        """

        gradients = np.zeros(X.shape)

        for b in range(0, len(X), batch_size):
 
            gradients[b:b + batch_size] = self.sess.run(self.grad, feed_dict={self.input_tensor: X[b:b + batch_size]})       	
        
        saliency = X*gradients

        if multiply_with_input:
            return saliency
        else:
            return gradients


class IntegratedGrad(object):
    """Implementaion of Integrated Gradient

        https://arxiv.org/pdf/1703.01365.pdf

        Example Usage:
        >>> attr_fn = IntegratedGrad(sess, input_tensor, output_tensor, 0)
        >>> integrated_grad = attr_fn(X, 'black')

    """
    def __init__(self, sess, input_tensor, output_tensor, target_class):
        """__Constructor__

        Arguments:
            sess {tf.Session} -- Tensorflow session
            input_tensor {tf.Tensor} -- Symbolic tensor of input node
            output_tensor {tf.Tensor} -- Symbolic tensor of pre-softmax output
            target_class {int} -- The class of interest to run explanation.
        """
        np.random.seed(1111)
        self.sess = sess
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.target_class = target_class

        self._define_ops()

    def _define_ops(self):
        """Add whatever operations you feel necessary into the computational
        graph.
        """

        self.y = self.output_tensor[:,self.target_class]
        self.grad = tf.gradients(xs=self.input_tensor, ys=self.y)[0]

    def __call__(self,
                 X,
                 baseline='black',
                 batch_size=16,
                 num_steps=50,
                 multiply_with_input=True):
        """__call__ forward computation to generate the integrated gradient

        Arguments:
            X {np.array} -- Input dataset

        Keyword Arguments:
            baseline {str} -- The baseline input. One of 'black', 'white',
                or 'random' (default: {'black'})
            batch_size {int} -- Batch size (default: {16})
            num_steps {int} -- resolution of using sum to approximate the
                integral (default: {50})
            multiply_with_input {bool} -- If True, return grad x input,
                else return grad (default: {True})

        Returns np.ndarray of the same shape as X.
        """
        if baseline=='random':
            b = np.random.normal(loc=0.0, scale=0.1, size=X.shape)

        
        elif baseline=='white':

            if X.max()>1:
                b = np.ones(X.shape)*255
            else:
                b = np.ones(X.shape)
        else:
            b = np.zeros(X.shape)

        scaled_inputs = np.array([b + (float(j)/num_steps)*(X-b) for j in range(1, num_steps+1)])
        scaled_inputs = np.swapaxes(scaled_inputs,0,1)

        gradients = []

        for i in range(scaled_inputs.shape[0]):
        	e= self.sess.run(self.grad,feed_dict= { self.input_tensor:scaled_inputs[i]})
        	e = np.average(e,axis=0)
        	gradients.append(e)

        	if i %10 == 0 and i!=0:
        		print(i)

        gradients = np.array(gradients)

        ig = (X - b)*gradients


        if multiply_with_input:
            return ig
        else:
            return gradients


if __name__ == '__main__':

    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'imagenet', sess)
    dataset = load_ImageNet()
    images = ['laska.png', 'camel.png']

    imgs = np.array([img_to_array(load_img(i, target_size=(224, 224))) for i in images])
    input_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    output_tensor = vgg.fc3l
    target = [356,354]
    for tar in  range(len(target)):
        print(target[tar])
        attr_fn1 = SaliencyMap(sess, input_tensor, output_tensor, target[tar])
        attr1= attr_fn1(np.expand_dims(imgs[tar],0))
        k = point_cloud(attr1, threshold=0)
        f = np.zeros((attr1.shape))
        f[:,:,:,0] = k
        f[:,:,:,1] = k
        f[:,:,:,2] = k

        x = binary_mask(np.expand_dims(imgs[tar],0), f,norm=True,threshold=0.2,blur=2,background=0.1)

        fig, ax = plt.subplots(1,3, figsize=(20, 3))
        fig.suptitle('Saliency Map')

        ax[0].imshow(imgs[tar]/255)
   
        ax[0].set_title('Input')

        ax[1].imshow(k[0])
        
        ax[1].set_title('Saliency Map')

        ax[2].imshow(x[0])
        
        ax[2].set_title('Saliency-masked Input')

       
        plt.savefig('part1q1'+str(tar)+'.png')
        plt.show()


    for tar in  range(len(target)):
        print(target[tar])
        attr_fn2 = IntegratedGrad(sess, input_tensor, output_tensor, target[tar])
        attr2 = attr_fn2(np.expand_dims(imgs[tar],0), 'black')
        k1 = point_cloud(attr2, threshold=0)
        f1 = np.zeros((attr2.shape))
        f1[:,:,:,0] = k1
        f1[:,:,:,1] = k1
        f1[:,:,:,2] = k1

        x1 = binary_mask(np.expand_dims(imgs[tar],0), f1,norm=True,threshold=0.2,blur=2,background=0.1)

        fig, ax = plt.subplots(1,3, figsize=(20, 3))
        fig.suptitle('Integrated Gradients')

        ax[0].imshow(imgs[tar]/255)
   
        ax[0].set_title('Input')

        ax[1].imshow(k1[0])
        
        ax[1].set_title('Saliency')

        ax[2].imshow(x1[0])
        
        ax[2].set_title('Saliency-masked Input')

       
        plt.savefig('part1q2'+str(tar)+'.png')
        plt.show()