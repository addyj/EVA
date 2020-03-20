## EVA-3 Phase-2 Session-1

### Text Embedding

---

#### Training Logs

Model where 8000 sample of 100 length is converted to 100 dimension embedding

Using Pre-Trained word embedding

```
Train on 8000 samples, validate on 10000 samples
Epoch 1/10
8000/8000 [==============================] - 2s 291us/step - loss: 0.6937 - acc: 0.5689 - val_loss: 0.6594 - val_acc: 0.5987
Epoch 2/10
8000/8000 [==============================] - 2s 246us/step - loss: 0.5905 - acc: 0.6874 - val_loss: 0.6153 - val_acc: 0.6494
Epoch 3/10
8000/8000 [==============================] - 2s 247us/step - loss: 0.4817 - acc: 0.7750 - val_loss: 0.6345 - val_acc: 0.6688
Epoch 4/10
8000/8000 [==============================] - 2s 245us/step - loss: 0.4132 - acc: 0.8095 - val_loss: 0.6044 - val_acc: 0.6975
Epoch 5/10
8000/8000 [==============================] - 2s 250us/step - loss: 0.3476 - acc: 0.8488 - val_loss: 0.6067 - val_acc: 0.7056
Epoch 6/10
8000/8000 [==============================] - 2s 249us/step - loss: 0.3071 - acc: 0.8685 - val_loss: 0.6485 - val_acc: 0.7069
Epoch 7/10
8000/8000 [==============================] - 2s 245us/step - loss: 0.2686 - acc: 0.8861 - val_loss: 0.6941 - val_acc: 0.6992
Epoch 8/10
8000/8000 [==============================] - 2s 246us/step - loss: 0.2198 - acc: 0.9127 - val_loss: 0.7702 - val_acc: 0.6982
Epoch 9/10
8000/8000 [==============================] - 2s 252us/step - loss: 0.1958 - acc: 0.9205 - val_loss: 1.5287 - val_acc: 0.5984
Epoch 10/10
8000/8000 [==============================] - 2s 253us/step - loss: 0.1626 - acc: 0.9350 - val_loss: 0.8947 - val_acc: 0.6941
```

#### Training and Validation curves with pre-trained word embedding

![img1](https://github.com/addyj/EVA/blob/master/EVA3_P2_Session1/T_V_A_WithPreTrain.png)

![img2](https://github.com/addyj/EVA/blob/master/EVA3_P2_Session1/T_V_L_WithPreTrain.png)

---

#### Evaluating pre-trained model

At Validation Accuracy Max : 70.69,  Training Accuracy : 86.85

```
25000/25000 [==============================] - 1s 49us/step
[0.9145144889330864, 0.6882]
```

Test Accuracy : 91.45

---

#### Training Logs

Model where 8000 sample of 100 length is converted to 100 dimension embedding

Without using Pre-Trained word embedding

```
Train on 8000 samples, validate on 10000 samples
Epoch 1/10
8000/8000 [==============================] - 5s 601us/step - loss: 0.5394 - acc: 0.7164 - val_loss: 0.3848 - val_acc: 0.8254
Epoch 2/10
8000/8000 [==============================] - 4s 544us/step - loss: 0.1408 - acc: 0.9521 - val_loss: 0.4345 - val_acc: 0.8164
Epoch 3/10
8000/8000 [==============================] - 4s 543us/step - loss: 0.0093 - acc: 0.9986 - val_loss: 0.5790 - val_acc: 0.8214
Epoch 4/10
8000/8000 [==============================] - 4s 545us/step - loss: 4.4753e-04 - acc: 1.0000 - val_loss: 0.7061 - val_acc: 0.8189
Epoch 5/10
8000/8000 [==============================] - 4s 551us/step - loss: 1.1627e-05 - acc: 1.0000 - val_loss: 0.8904 - val_acc: 0.8149
Epoch 6/10
8000/8000 [==============================] - 4s 544us/step - loss: 7.7427e-07 - acc: 1.0000 - val_loss: 0.9267 - val_acc: 0.8209
Epoch 7/10
8000/8000 [==============================] - 4s 545us/step - loss: 1.1973e-07 - acc: 1.0000 - val_loss: 0.9546 - val_acc: 0.8206
Epoch 8/10
8000/8000 [==============================] - 4s 543us/step - loss: 1.1056e-07 - acc: 1.0000 - val_loss: 0.9677 - val_acc: 0.8203
Epoch 9/10
8000/8000 [==============================] - 4s 543us/step - loss: 1.1008e-07 - acc: 1.0000 - val_loss: 0.9759 - val_acc: 0.8204
Epoch 10/10
8000/8000 [==============================] - 4s 547us/step - loss: 1.0990e-07 - acc: 1.0000 - val_loss: 0.9823 - val_acc: 0.8191
```

#### Training and Validation curves without pre-trained word embedding

![img3](https://github.com/addyj/EVA/blob/master/EVA3_P2_Session1/T_V_A_WithoutPreTrain.png)

![img4](https://github.com/addyj/EVA/blob/master/EVA3_P2_Session1/T_V_L_WithoutPreTrain.png)

Without PreTrained 

At Validation Accuracy Max : 82.54,  Training Accuracy : 71.64
