import numpy as np
import tensorflow as tf
import hw3_utils
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
from hw3_infl import *
from hw3_attribution import *
from hw3_evaluation import *


if __name__ == '__main__':
    data = hw3_utils.load_ImageNet()

    # print(data.X.shape)
    # print(data.Y.shape)
    # print(data.Pred.shape)
    # print(data.Idx.shape)

    # >>> Your code for written exercises here <<<
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'imagenet', sess)
    dataset = load_ImageNet()
    
    input_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    output_tensor = vgg.fc3l
    internal_layer = vgg.fc1
    imgs = np.array([dataset[0][dataset[-1]==1504][0],dataset[0][dataset[-1]==11436][0],dataset[0][dataset[-1]==11691][0],dataset[0][dataset[-1]==16075][0],dataset[0][dataset[-1]==111][0]])
    target = [386,354,397,734,284]

    #imgs = np.array([dataset[0][dataset[-1]==11993][0],dataset[0][dataset[-1]==1338][0],dataset[0][dataset[-1]==10857][0],dataset[0][dataset[-1]==11055][0],dataset[0][dataset[-1]==27643][0]])
    #target = [801,354,178,511,407]

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

       
        plt.savefig('A_in'+str(target[tar])+'.png')
        plt.show()


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
        plt.savefig('B_in'+str(target[tar])+'.png')
        plt.show()



        attr_fn3 = InflDirected(sess, input_tensor, internal_layer,output_tensor, qoi=lambda out: out[target[tar]])
        neuron=attr_fn3.dis_influence(dataset[0], batch_size=100)
        attr3 = attr_fn3.expert_attribution(neuron,np.expand_dims(imgs[tar],0))
        k3 = point_cloud(attr3, threshold=0)
        f3 = np.zeros((attr3.shape))
        f3[:,:,:,0] = k3
        f3[:,:,:,1] = k3
        f3[:,:,:,2] = k3

        x3 = binary_mask(np.expand_dims(imgs[tar],0), f3,norm=True,threshold=0.2,blur=2,background=0.1)

        fig, ax = plt.subplots(1,3, figsize=(20, 3))
        fig.suptitle('Influence Directed Explanations')

        ax[0].imshow(imgs[tar]/255)
   
        ax[0].set_title('Input')

        ax[1].imshow(k3[0])
        
        ax[1].set_title('Expert at fc_1')

        ax[2].imshow(x3[0])
        
        ax[2].set_title('Visualizations')
        plt.savefig('C_in'+str(target[tar])+'.png')
        plt.show()
       