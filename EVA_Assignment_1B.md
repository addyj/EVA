#### _ADITYA JINDAL_

#### _BATCH M6_

#### _ASSIGNMENT 1B_

#### EVA

------

### What are Channels and Kernels ?

### Kernels

This can be constructed as a matrix of floating points (aka weights) whose amplitude and pattern helps us get the desired outputs when convoluted with the input image. To get a new feature we need to combine many input features and this number is directly related to the kernel size. It is used for extracting various image data like edges, gradients or diminishing and enhancing features by blurring, sharpening, unsharpening, embossing etc. The number of kernels used is always equal to the number of input channels. A kernel can be both symmetrical or unsymmetrical. This is also known as filters or feature extractors.

### Feature Maps/Channels

These are the input/output image data which you get from the convolution of input image/channel and kernel in every layer. We can also divide our image into as many maps/channels as we want (i.e. the separations can be based on colours, edges, textures etc.). Every channel has its own feature map. In general term, we say its the features and their location in the image. From convolution, after finding features based on their activation we store them in these maps. Taking different feature combination can also point in the existence of a complex feature also. The number of maps/channels we get is the same as the number of Filters/Kernels. Also called as activation Maps as ultimately it only stores the activation values of different parts of the input image.

------

### Why should we only use 3x3 Kernels ?

- Computationally faster as uses less number of weights.
- Give us  a choice of having more number of layers as less weights.
- In turn more layers helps the model learn more complexities.
- The drawback is that it need more memory for storage as more back propagation is required.
- All other kernels are ultimately made up from 3x3 as building blocks (e.g two 3x3 kernel makes up one 5x5 kernel).
- We can also use even sized kernels, but we generally don't. Odd filters are preferred because of symmetric nature. All the layer pixels are symmetrically around the output pixel. If we don't take this symmetry into consideration, we will have to account for distortions across the layers.

---

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 ?

As we know that convolving on nxn image with a 3x3 kernel reduces the dimension of the image by 2 , hence giving us an output of (n-2)x(n-2). This changes with the size of the kernel.

Therefore by calculation we will have around 199/2 = 99.5 layers.

By logic we can't have half a layer so we have to use some other operation (like pooling , padding etc.) to reach exactly 1x1, where we will be seeing the whole image in support to the global receptive field. 

Therefore __at-most__ 99 or 100 times we need to perform 3x3 convolution.

Showing the calculations:  

__[Original res.]199x199__ ==__(3x3)[Kernel]__==> 197x197[Channel] ==3x3==> 195x195 ==3x3==> 193x193 ==3x3==> 191x191 ==3x3==> 189x189 ==3x3==> 187x187 ==3x3==> 185x185 ==3x3==> 183x183 ==3x3==> 181x181 ==3x3==> 179x179 ==3x3==> 177x177 ==3x3==> 175x175 ==3x3==> 173x173 ==3x3==> 171x171 ==3x3==> 169x169 ==3x3==> 167x167 ==3x3==> 165x165 ==3x3==> 163x163 ==3x3==> 161x161 ==3x3==> 159x159 ==3x3==> 157x157 ==3x3==> 155x155 ==3x3==> 153x153 ==3x3==> 151x151 ==3x3==> 149x149 ==3x3==> 147x147 ==3x3==> 145x145 ==3x3==> 143x143 ==3x3==> 141x141 ==3x3==> 139x139 ==3x3==> 137x137 ==3x3==> 135x135 ==3x3==> 133x133 ==3x3==> 131x131 ==3x3==> 129x129 ==3x3==> 127x127 ==3x3==> 125x125 ==3x3==> 123x123 ==3x3==> 121x121 ==3x3==> 119x119 ==3x3==> 117x117 ==3x3==> 115x115 ==3x3==> 113x113 ==3x3==> 111x111 ==3x3==> 109x109 ==3x3==> 107x107 ==3x3==> 105x105 ==3x3==> 103x103 ==3x3==> 101x101 ==3x3==> 99x99 ==3x3==> 97x97 ==3x3==> 95x95 ==3x3==> 93x93 ==3x3==> 91x91 ==3x3==> 89x89 ==3x3 ==> 87x87 ==3x3==> 85x85 ==3x3==> 83x83 ==3x3==> 81x81 ==3x3==> 79x79 ==3x3==> 77x77 ==3x3==> 75x75 ==3x3==> 73x73 ==3x3==> 71x71 ==3x3==> 69x69 ==3x3==> 67x67 ==3x3==> 65x65 ==3x3==> 63x63 ==3x3==> 61x61 ==3x3==> 59x59 ==3x3==> 57x57 ==3x3==> 55x55 ==3x3==> 53x53 ==3x3==> 51x51 ==3x3==> 49x49 ==3x3==> 47x47 ==3x3==> 45x45 ==3x3==> 43x43 ==3x3==> 41x41 ==3x3==> 39x39 ==3x3==> 37x37 ==3x3==> 35x35 ==3x3==> 33x33 ==3x3==> 31x31 ==3x3==> 29x29 ==3x3==> 27x27 ==3x3==> 25x25 ==3x3==> 23x23 ==3x3==> 21x21 ==3x3==> 19x19 ==3x3==> 17x17 ==3x3==> 15x15 ==3x3==> 13x13 ==3x3==> 11x11 ==3x3==> 9x9 ==3x3==> 7x7 ==3x3==> 5x5 ==3x3==> 3x3 ==3x3==> __1x1 (pixel which sees the whole Image)__

Therefore is took 99 layers to reach 1x1.







   











---

