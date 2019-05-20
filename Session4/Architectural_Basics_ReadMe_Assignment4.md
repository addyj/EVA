### Aditya Jindal

#### Batch M6

### Assignment 4

---

## *Thought Process on Neural Network Architecture*

##### The basis of my thinking is purely hit and trial. Don't know if its the correct way of thinking. After knowing what i have to achieve as a target w.r.t Validation accuracy, Total model parameters , and Epochs, i try to build a network and keep tweaking it with the concepts i know, until it fits like a puzzle piece.

---

##### My thinking order mostly is:

- **Layers** to get the exact number of prediction classes.
- Also declaring layers such that the required Receptive Field is met.
- Using **3x3 Convolutions** only.
- **Max Pooling** to reduce Channel size and increase Receptive Field.
- Keeping the **Max Pool layer** at least 2-3 convolutions after the first layer, and at least 2-3 layers before the prediction layer, as we need our model to at learn edges and gradients properly before we discard the channel dimensions.
- **1x1 Convolutions** to only decrease channels.
- Relu **Activation's**, not to use with last layer, to pass unbiased information to prediction layer.
- **Softmax layer** to have presentable distinction between predicted classes. 
- Train with **validation checks**, so that we can keep track of over-fitting and judge how our model is doing.

##### If the above doesn't result good, i move forward with:

- Changing/Increasing **Max-pools position/numbers**.

- Adding/Removing **Point-wise Convolutions** to adjust the architecture.

- Adding **Batch Normalisation** to equate amplitude ranges of the images in a batch.

- Not adding Batch Normalisation to the last layer as we want to pass on the unaltered information for prediction. Also, as experimented in papers I add Batch Normalisation before convolution layer.

- **Image Normalisation**.

- **Changing kernel** channels to tweak parameter numbers based on our GPU memory and image data we can decide on how narrow or wide we want our network to be.
- When the relevant portion of the image is smaller that the total size of the image we don't need to convolve on it meticulously. We can do convolutions with a **bigger kernel size** as we don't have much relevant information to lose. Another alternative is using Global Average Pool Layer. At this stage by doing normal convolution we are using the same pixels repeatedly for convolution which are not leading us to any new information (do it before prediction layer).

##### Now onto the part on how can we increase our validation accuracy without over-fitting and any other complications such as OOM
- Increasing **Epochs**, I do this when I see almost a continuous kind of a rise in the validation accuracy
- Increasing **Batch size**, with enough GPU memory we can keep a good size, which help our epoch run faster, as its looking at more images in a single iteration to give a good back propagation.
- Using **Dropout**, when the gap between validation accuracy and train accuracy increase. Prefer using it as an Image Augmentation instead of using it as a layer.
- Though it help us in reducing the gap between validation accuracy and train accuracy (i.e. Over-fitting and Under-fitting), using this we might lose essential kernel which were going to help us in giving a better prediction. 
- Using it after every convolution layer with the discard rate less than 0.3, but not with the last (prediction) layer as we don't want our prediction to be incorrect due to less information. 
- Changing **Learning Rates** using different techniques (like CLR, ReduceLRonPlateau), so that we can avoid getting stuck in a minima of the cost curve.
- Using **LR schedule** to change learning rate also has its own advantage. We can change the learning rate based on a formula depending on any factor like accuracy or epoch count.

- Changing **Optimiser** (prefer using Adam, sometimes SGD with nestrov and momentum).
- We can know that if our network is not going well by just looking at the first few epoch runs. The factors like epoch run time, gap between validation accuracy and train accuracy, losses, all together can greatly help us in **judging the potential** of our model.

  ###			                         		                               *Think $ Build $ Train $ Repeat*