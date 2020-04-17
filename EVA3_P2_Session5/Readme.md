Group Members-
1. Nishad Rajmalwar (nrajmalwar@gmail.com)
2. Aditya Jindal (adityajindal4@gmail.com)

# Model Training on MNIST dataset using Pytorch

The model has following features-

1. Total parameters - 14,624
2. uses dropout of 0.1 
3. uses batch normalization
4. uses randomrotate transform of +/- 10 degress
5. uses StepLR with step size = 6 and gamma = 0.1
6. achieves 99.3% test accuracy

# Model Summary - 
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
       BatchNorm2d-2           [-1, 16, 26, 26]              32
              ReLU-3           [-1, 16, 26, 26]               0
            Conv2d-4           [-1, 24, 24, 24]           3,456
       BatchNorm2d-5           [-1, 24, 24, 24]              48
           Dropout-6           [-1, 24, 24, 24]               0
              ReLU-7           [-1, 24, 24, 24]               0
         MaxPool2d-8           [-1, 24, 12, 12]               0
            Conv2d-9           [-1, 16, 12, 12]             384
      BatchNorm2d-10           [-1, 16, 12, 12]              32
             ReLU-11           [-1, 16, 12, 12]               0
           Conv2d-12           [-1, 24, 10, 10]           3,456
      BatchNorm2d-13           [-1, 24, 10, 10]              48
          Dropout-14           [-1, 24, 10, 10]               0
             ReLU-15           [-1, 24, 10, 10]               0
        MaxPool2d-16             [-1, 24, 5, 5]               0
           Conv2d-17             [-1, 16, 5, 5]             384
      BatchNorm2d-18             [-1, 16, 5, 5]              32
             ReLU-19             [-1, 16, 5, 5]               0
           Conv2d-20             [-1, 28, 3, 3]           4,032
      BatchNorm2d-21             [-1, 28, 3, 3]              56
          Dropout-22             [-1, 28, 3, 3]               0
             ReLU-23             [-1, 28, 3, 3]               0
           Conv2d-24             [-1, 10, 1, 1]           2,520
================================================================
Total params: 14,624
Trainable params: 14,624
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.84
Params size (MB): 0.06
Estimated Total Size (MB): 0.90
----------------------------------------------------------------
```
# Model Plot - 

![alt text](https://github.com/nrajmalwar/Project-1/blob/master/Images/plot_P2S5.png)

# Model achieves highest test accuracy of 99.37% at Epoch 10
