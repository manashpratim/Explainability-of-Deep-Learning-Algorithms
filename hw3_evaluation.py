import tensorflow as tf
import numpy as np
from hw3_utils import binary_mask, point_cloud
import matplotlib.pyplot as plt
from hw3_utils import *
from hw3_infl import *
from hw3_attribution import *


def AverageDrop(sess, attr_map, input,
                input_tensor, output_tensor,
                target_class):
    """Implementation of Average Drop %

    https://arxiv.org/abs/1710.11063

    Arguments:
        sess {tf.Session} -- tensorflow session
        attr_map {np.ndarray} -- NxHxW, the attribution maps for input
        input {np.ndarray} -- NxHxWxC, the input dataset
        input_tensor {tf.Tensor} -- Symbolic tensor of input node
        output_tensor {tf.Tensor} -- Symbolic tensor of pre-softmax output
        target_class: {int} -- Class of Interest

    Returns:
        float -- average drop % score
    """
    score = None
    # >>> Your code starts here <<<

    #declaring the symbolic tensor for the score of the target class
    y = output_tensor[:,target_class]
    
    #turning all the positive values of attribute map to 1
    attr_map[attr_map>0] = 1

    #creating a mask equal to the dimension of the input
    masked  = np.zeros(input.shape)
    masked[:,:,:,0] = attr_map
    masked[:,:,:,1] = attr_map
    masked[:,:,:,2] = attr_map 
    
    #the mask keeps only the pixels in input whose pixel attribution score in A is positive
    masked_input = masked*input
    
    #f_c(x_i) 
    f1 = sess.run(y, feed_dict={input_tensor: input})
 
    #f_c(M_A_(x_i))
    f2 = sess.run(y, feed_dict={input_tensor: masked_input})
    #Computing the AD(D)
    score = np.sum(np.maximum(f1-f2,0)/f1)/len(input)

    
    #Write-up version. Here, c = arg max(f(x_i)) 
    
    #declaring the symbolic tensor for the score of the target class
    """y = output_tensor
    
    #turning all the positive values of attribute map to 1
    attr_map[attr_map>0] = 1

    #creating a mask equal to the dimension of the input
    masked  = np.zeros(input.shape)
    masked[:,:,:,0] = attr_map
    masked[:,:,:,1] = attr_map
    masked[:,:,:,2] = attr_map 
    
    #the mask keeps only the pixels in input whose pixel attribution score in A is positive
    masked_input = masked*input
    
    #f(x_i)
    f1 = sess.run(y, feed_dict={input_tensor: input})
    #f_c(x_i) where c = arg max(f(x_i)) 
    f1 = np.max(f1,axis=1)
    #f(M_A_(x_i))
    f2 = sess.run(y, feed_dict={input_tensor: masked_input})
    #f_c(M_A_(x_i)) where c = arg max(f(M_A_(x_i))) 
    f2 = np.max(f2,axis=1)
    #Computing the AD(D)
    score = np.sum(np.maximum(f1-f2,0)/f1)/len(input)"""   

    # >>> Your code ends here <<<
    return score*100




def N_Ord(sess, attr_map, input,
          input_tensor, output_tensor,
          target_class):
    """Implementation of Necessity Ordering

    https://arxiv.org/abs/2002.07985

    Arguments:
        sess {tf.Session} -- tensorflow session
        attr_map {np.ndarray} -- HxW, the attribution map for input
        input {np.ndarray} -- HxWxC, the input image
        input_tensor {tf.Tensor} -- Symbolic tensor of input node
        output_tensor {tf.Tensor} -- Symbolic tensor of pre-softmax output
        target_class: {int} -- Class of Interest

    Returns:
        float -- Necessity Drop score
    """
    score = None
    # >>> Your code starts here <<<

    l = attr_map.flatten()
    N = np.count_nonzero(l)    #No. of positive features

    y = output_tensor[:,target_class]       #Symbolic tensor for pre_softmax score
    x = np.zeros(input.shape)
    f2 = sess.run(y, feed_dict={input_tensor: np.expand_dims(x,0)})    #baseline score

    total = 0
    p = 0
    
    for i in range(0,N,1000):

        l[np.argsort(l)[::-1][:1000]] = 0                #turning M highest values of attribute map to 0

        z = np.zeros(input.shape)                        #creating a mask 
        z[:,:,0] = l.reshape(224,224)
        z[:,:,1] = l.reshape(224,224)
        z[:,:,2] = l.reshape(224,224)
        z[z>0] = 1                                       #positive values of attribute map are turned to 1
        inp = input*z                                    #masked input
        
        f1 = sess.run(y, feed_dict={input_tensor: np.expand_dims(inp,0)})           #output score

        t =  (N - np.count_nonzero(l))- p                                           #some intermediate steps to keep track of number of features removed in a pass
        p = N - np.count_nonzero(l)

        total = total + np.maximum(f1-f2,0)*t                                       #score of this pass is multiplied by the number of features removed as M of the terms at a time achieve the same value

    score = total/(N+1)
        
        

    # >>> Your code ends here <<<
    return score


if __name__ == '__main__':
    #>>> Your code for written exercises here <<<
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'imagenet', sess)
    dataset = load_ImageNet()
    images = ['laska.png', 'camel.png']
    imgs = np.array([img_to_array(load_img(i, target_size=(224, 224))) for i in images])

    input_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    output_tensor = vgg.fc3l
    internal_layer = vgg.fc1

    #positive samples
    imgs = np.array([dataset[0][dataset[-1]==1504][0],dataset[0][dataset[-1]==11436][0],dataset[0][dataset[-1]==11691][0],dataset[0][dataset[-1]==16075][0],dataset[0][dataset[-1]==111][0]])
    target = [386,354,397,734,284]
    
    #negative samples
    #imgs = np.array([dataset[0][dataset[-1]==11993][0],dataset[0][dataset[-1]==1338][0],dataset[0][dataset[-1]==10857][0],dataset[0][dataset[-1]==11055][0],dataset[0][dataset[-1]==27643][0]])
    #target = [801,354,178,511,407]

    for tar in  range(len(target)):
        print(target[tar])
    
        attr_fn1 = SaliencyMap(sess, input_tensor, output_tensor, target[tar])
        attr1 = attr_fn1(np.expand_dims(imgs[tar],0))
        #attr1 = attr_fn1(dataset[0][dataset[1] == target])
        print('Saliency Map Done!!!')

        attr_fn2 = IntegratedGrad(sess, input_tensor, output_tensor, target[tar])
        attr2 = attr_fn2(np.expand_dims(imgs[tar],0), 'black')
        print('IntegratedGrad Done!!!')
     
        attr_fn3 = InflDirected(sess, input_tensor, internal_layer,output_tensor, qoi=lambda out: out[target[tar]])
        #neuron=attr_fn3.dis_influence(dataset[0][dataset[1] == target], batch_size=16)
        neuron=attr_fn3.dis_influence(dataset[0], batch_size=50)
        #attr3 = attr_fn3.expert_attribution(neuron,dataset[0][dataset[1] == target],batch_size=16,multiply_with_input=True)
        attr3 = attr_fn3.expert_attribution(neuron,np.expand_dims(imgs[tar],0),batch_size=16,multiply_with_input=True)
        print('InflDirected Done!!!')

        #score1 = AverageDrop(sess, point_cloud(attr1,threshold=0), dataset[0][dataset[1] == target[tar]],input_tensor, output_tensor,target[tar])
        #score2 = AverageDrop(sess, point_cloud(attr2,threshold=0), dataset[0][dataset[1] == target[tar]],input_tensor, output_tensor,target[tar])
        #score3 = AverageDrop(sess, point_cloud(attr3,threshold=0), dataset[0][dataset[1] == target[tar]],input_tensor, output_tensor,target[tar])
        score1 = N_Ord(sess, point_cloud(attr1,threshold=0)[0],imgs[tar] ,input_tensor, output_tensor,target[tar])
        score2 = N_Ord(sess, point_cloud(attr2,threshold=0)[0],imgs[tar] ,input_tensor, output_tensor,target[tar])
        score3 = N_Ord(sess, point_cloud(attr3,threshold=0)[0],imgs[tar] ,input_tensor, output_tensor,target[tar])

        print(score1,score2,score3)

