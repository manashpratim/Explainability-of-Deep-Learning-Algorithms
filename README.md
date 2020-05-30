# Explainability-of-Deep-Learning-Algorithms
What factors influence the predictions of Deep learning Algorithms?
## Overview

The goal of an attribution method is to determine which parts of an input image a trained model uses to predict the class of the input image. If those parts of the image is perturbed then the model will fail to correctly classify the image.

In this project, I have implemented a VGG 16 model and the following attribution methods:

1. Saliency map [1] (hw3_attribution.py), 
2. Integrated Gradients [2] (hw3_attribution.py), 
3. Influence-directed Explanations [3] (hw3_infl.py) 
4. Representer Points [4] (hw4_part3.py) 

Then, I have compared these methods using visual comparisons as well as quantitative metrics like Average % Drop and Necessity Ordering [5] (hw3_evaluation.py)

## Results

### Saliency Map

<p align="center">
  <img width="550" height="300" src="https://github.com/manashpratim/Explainability-of-Deep-Learning-Algorithms/blob/master/sm.PNG">
</p>

### Integrated Gradients

<p align="center">
  <img width="550" height="300" src="https://github.com/manashpratim/Explainability-of-Deep-Learning-Algorithms/blob/master/ig.PNG">
</p>

### Influence-directed Explanations

<p align="center">
  <img width="550" height="300" src="https://github.com/manashpratim/Explainability-of-Deep-Learning-Algorithms/blob/master/ide.PNG">
</p>

### Representer Points

<p align="center">
  <img width="550" height="300" src="https://github.com/manashpratim/Explainability-of-Deep-Learning-Algorithms/blob/master/ip.PNG">
</p>

<p align="center">
  <img width="550" height="300" src="https://github.com/manashpratim/Explainability-of-Deep-Learning-Algorithms/blob/master/fp.PNG">
</p>

<p align="center">
  <img width="300" height="200" src="https://github.com/manashpratim/Explainability-of-Deep-Learning-Algorithms/blob/master/class.PNG">
</p>

## References
[1] David Baehrens, Timon Schroeter, Stefan Harmeling, Motoaki Kawanabe, Katja Hansen, and Klaus- Robert Mueller. How to explain individual classification decisions, 2009.

[2] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. arXiv preprint arXiv:1703.01365, 2017.

[3] Klas Leino, Linyi Li, Shayak Sen, Anupam Datta, and Matt Fredrikson. Inuence-directed explanations for deep convolutional networks. arXiv preprint arXiv:1802.03788, 2018.

[4] Chih-Kuan Yeh, Joon Kim, Ian En-Hsu Yen, and Pradeep K Ravikumar. Representer point selection for explaining deep neural networks. In Advances in Neural Information Processing Systems, pages 9291–9301, 2018.

[5] Zifan Wang, PiotrPiotr Mardziel, Anupam Datta, and Matt Fredrikson. Interpreting interpretations: Organizing attribution methods by criteria, 2020.

**Note: This project is part of my Homeworks. Current CMU students please refrain from going through the codes.**


