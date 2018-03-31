# CNN-image-classification :eyeglasses: :dart:
Designing and tuning a convolutional neural network for image classification

1. Experimentation with the topology of the neural network and its hyper-parameters to evaluate the effects on the model performance
2. Building a neural network for the classification of images (fashion-MNIST dataset is used --> made available by Zalando Research)
3. Design a network using convolutional layers, possibly combined with pooling layers for the SCT dataset
4. The end objective is to design a network using convolutional layers, possibly combined with pooling layers and to tune the parameters, #layers, layer-sizes et cetera to achieve a relatively good performance for the Cifar-10 dataset containing 60,000 32x32 colored images (10 classes)

Techniques/approaches/update-functions etc. that are used in this project:
- [x] Mean-subtraction normalization
- [x] Gradient Descent with Momentum
- [x] Gradient Descent Nesterov Momentum
- [x] L2-weight decay
- [x] Ada-Delta
- [x] DropOut regularization

For more information, please visit:
[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)

Datasets/resources that were used:
- [Zalando Research: Fashion-MNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
- [Canadian Institute For Advanced Research: CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html/)
