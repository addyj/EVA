## Aditya Jindal
## Batch M6
## Assignment 7A
## *Receptive Field Calculations*

#### J<sub>Out</sub> = J<sub>In</sub> * S   

####  J<sub>In</sub> = J<sub>Out</sub>

#### R<sub>Out</sub> = R<sub>In</sub> + (K-1) * J<sub>In</sub> 

#### Aim:  To prove that Receptive Field is 224

---

### Architecture for GoogleNet

---

#### Layer_1: Convolution 7×7 | Stride = 2 | Output = 112×112×64 

J<sub>in</sub> = 1,  J<sub>out</sub> = 2, K = 7, R<sub>in</sub> = 1, R<sub>out</sub> = **7**

---

#### Layer_2: Max Pool 3×3 | Stride = 2 | Output = 56×56×64 

J<sub>in</sub> = 2,  J<sub>out</sub> = 4, K = 3, R<sub>in</sub> = 7, R<sub>out</sub> = **11**

---

#### Layer_31: Convolution 1×1 | Stride = 1 | Output = 56×56×64 

J<sub>in</sub> = 4,  J<sub>out</sub> = 4, K = 1, R<sub>in</sub> = 11, R<sub>out</sub> = **11**

#### Layer_32: Convolution 3×3 | Stride = 1 | Output = 56×56×192 

J<sub>in</sub> = 4,  J<sub>out</sub> = 4, K = 3, R<sub>in</sub> = 11, R<sub>out</sub> = **19**

---

#### Layer_4: Max Pool 3×3 | Stride = 2 | Output = 28×28×192 

J<sub>in</sub> = 4,  J<sub>out</sub> = 8, K = 3, R<sub>in</sub> = 19, R<sub>out</sub> = **27**

---

#### Layer_5: Inception 3A

##### (Grouped Layers)

#### Layer_51: Convolution 1×1 | Stride = 1 | Output = 28×28×64

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 1, R<sub>in</sub> = 27, R<sub>out</sub> = **27**

---

#### Layer_52: Convolution 1×1 | Stride = 1 | Output = 28×28×96

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 1, R<sub>in</sub> = 27, R<sub>out</sub> = **27**

#### Layer_53: Convolution 3×3 | Stride = 1 | Output = 28×28×128

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 3, R<sub>in</sub> = 27, R<sub>out</sub> = **43**

---

#### Layer_54: Convolution 1×1 | Stride = 1 | Output = 28×28×16 

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 1, R<sub>in</sub> = 27, R<sub>out</sub> = **27**

#### Layer_55: Convolution 5×5 | Stride = 1 | Output = 28×28×32

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 5, R<sub>in</sub> = 27, R<sub>out</sub> = **59**

---

#### Layer_56: Max Pool 3×3 | Stride = 1 | Output = 28×28×192 

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 3, R<sub>in</sub> = 27, R<sub>out</sub> = **43**

#### Layer_57: Convolution 1×1 | Stride = 1 | Output = 28×28×32 

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 1, R<sub>in</sub> = 43, R<sub>out</sub> = **43**

---

#### Layer_5: Concatenation | Output = 28×28×256

R<sub>out</sub> [max] = **59**

---

#### Layer_6: Inception 3B

##### (Grouped Layers)

#### Layer_61: Convolution 1×1 | Stride = 1 | Output = 28×28×128

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 1, R<sub>in</sub> = 59, R<sub>out</sub> = **59**

------

#### Layer_62: Convolution 1×1 | Stride = 1 | Output = 28×28×128

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 1, R<sub>in</sub> = 59, R<sub>out</sub> = **59**

#### Layer_63: Convolution 3×3 | Stride = 1 | Output = 28×28×192

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 3, R<sub>in</sub> = 59, R<sub>out</sub> = **75**

------

#### Layer_64: Convolution 1×1 | Stride = 1 | Output = 28×28×32

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 1, R<sub>in</sub> = 59, R<sub>out</sub> = **59**

#### Layer_65: Convolution 5×5 | Stride = 1 | Output = 28×28×96

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 5, R<sub>in</sub> = 59, R<sub>out</sub> = **91**

------

#### Layer_66: Max Pool 3×3 | Stride = 1 | Output = 28×28×256 

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 3, R<sub>in</sub> = 59, R<sub>out</sub> = **75**

#### Layer_67: Convolution 1×1 | Stride = 1 | Output = 28×28×64 

J<sub>in</sub> = 8,  J<sub>out</sub> = 8, K = 1, R<sub>in</sub> = 75, R<sub>out</sub> = **75**

------

#### Layer_6: Concatenation | Output = 28×28×480

R<sub>out</sub> [max] = **91**

---

#### Layer_7: Max Pool 3×3 | Stride = 2 | Output = 14×14×480 

J<sub>in</sub> = 8,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 91, R<sub>out</sub> = **107**

------

#### Layer_8: Inception 4A

##### (Grouped Layers)

#### Layer_81: Convolution 1×1 | Stride = 1 | Output = 14×14×192

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 107, R<sub>out</sub> = **107**

------

#### Layer_82: Convolution 1×1 | Stride = 1 | Output = 14×14×96

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 107, R<sub>out</sub> = **107**

#### Layer_83: Convolution 3×3 | Stride = 1 | Output = 14×14×208

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 107, R<sub>out</sub> = **139**

------

#### Layer_84: Convolution 1×1 | Stride = 1 | Output = 14×14×16

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 107, R<sub>out</sub> = **107**

#### Layer_85: Convolution 5×5 | Stride = 1 | Output = 14×14×48

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 5, R<sub>in</sub> = 107, R<sub>out</sub> = **171**

------

#### Layer_86: Max Pool 3×3 | Stride = 1 | Output = 14×14×480

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 107, R<sub>out</sub> = **139**

#### Layer_87: Convolution 1×1 | Stride = 1 | Output = 14×14×64 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 139, R<sub>out</sub> = **139**

------

#### Layer_8: Concatenation | Output = 14×14×512

R<sub>out</sub> [max] = **171**

---

#### Layer_9: Inception 4B

##### (Grouped Layers)

#### Layer_91: Convolution 1×1 | Stride = 1 | Output = 14×14×160

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 171, R<sub>out</sub> = **171**

------

#### Layer_92: Convolution 1×1 | Stride = 1 | Output = 14×14×112

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 171, R<sub>out</sub> = **171**

#### Layer_93: Convolution 3×3 | Stride = 1 | Output = 14×14×224

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 171, R<sub>out</sub> = **203**

------

#### Layer_94: Convolution 1×1 | Stride = 1 | Output = 14×14×24

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 171, R<sub>out</sub> = **171**

#### Layer_95: Convolution 5×5 | Stride = 1 | Output = 14×14×64

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 5, R<sub>in</sub> = 171, R<sub>out</sub> = **235**

------

#### Layer_96: Max Pool 3×3 | Stride = 1 | Output = 14×14×512 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 171, R<sub>out</sub> = **203**

#### Layer_97: Convolution 1×1 | Stride = 1 | Output = 14×14×64 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 203, R<sub>out</sub> = **203**

------

#### Layer_9: Concatenation | Output = 14×14×512

R<sub>out</sub> [max] = **235**

------

#### Layer_10: Inception 4C

##### (Grouped Layers)

#### Layer_101: Convolution 1×1 | Stride = 1 | Output = 14×14×128

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 235, R<sub>out</sub> = **235**

------

#### Layer_102: Convolution 1×1 | Stride = 1 | Output = 14×14×128

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 235, R<sub>out</sub> = **235**

#### Layer_103: Convolution 3×3 | Stride = 1 | Output = 14×14×256

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 235, R<sub>out</sub> = **267**

------

#### Layer_104: Convolution 1×1 | Stride = 1 | Output = 14×14×24

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 235, R<sub>out</sub> = **235**

#### Layer_105: Convolution 5×5 | Stride = 1 | Output = 14×14×64

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 5, R<sub>in</sub> = 235, R<sub>out</sub> = **299**

------

#### Layer_106: Max Pool 3×3 | Stride = 1 | Output = 14×14×512 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 235, R<sub>out</sub> = **267**

#### Layer_107: Convolution 1×1 | Stride = 1 | Output = 14×14×64 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 267, R<sub>out</sub> = **267**

------

#### Layer_10: Concatenation | Output = 14×14×512 

R<sub>out</sub> [max] = **299**

---

#### Layer_11: Inception 4D

##### (Grouped Layers)

#### Layer_111: Convolution 1×1 | Stride = 1 | Output = 14×14×112

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 299, R<sub>out</sub> = **299**

------

#### Layer_112: Convolution 1×1 | Stride = 1 | Output = 14×14×144

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 299, R<sub>out</sub> = **299**

#### Layer_113: Convolution 3×3 | Stride = 1 | Output = 14×14×288

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 299, R<sub>out</sub> = **331**

------

#### Layer_114: Convolution 1×1 | Stride = 1 | Output = 14×14×32

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 299, R<sub>out</sub> = **299**

#### Layer_115: Convolution 5×5 | Stride = 1 | Output = 14×14×64

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 5, R<sub>in</sub> = 299, R<sub>out</sub> = **363**

------

#### Layer_116: Max Pool 3×3 | Stride = 1 | Output = 14×14×512 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 299, R<sub>out</sub> = **331**

#### Layer_117: Convolution 1×1 | Stride = 1 | Output = 14×14×64 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 331, R<sub>out</sub> = **331**

------

#### Layer_11: Concatenation | Output = 14×14×528 

R<sub>out</sub> [max] = **363** 

------

#### Layer_12: Inception 4E

##### (Grouped Layers)

#### Layer_121: Convolution 1×1 | Stride = 1 | Output = 14×14×256

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 363, R<sub>out</sub> = **363**

------

#### Layer_122: Convolution 1×1 | Stride = 1 | Output = 14×14×160

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 363, R<sub>out</sub> = **363**

#### Layer_123: Convolution 3×3 | Stride = 1 | Output = 14×14×320

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 363, R<sub>out</sub> = **395**

------

#### Layer_124: Convolution 1×1 | Stride = 1 | Output = 14×14×32

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 363, R<sub>out</sub> = **363**

#### Layer_125: Convolution 5×5 | Stride = 1 | Output = 14×14×128

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 5, R<sub>in</sub> = 363, R<sub>out</sub> = **427**

------

#### Layer_126: Max Pool 3×3 | Stride = 1 | Output = 14×14×528 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 3, R<sub>in</sub> = 363, R<sub>out</sub> = **395**

#### Layer_127: Convolution 1×1 | Stride = 1 | Output = 14×14×128 

J<sub>in</sub> = 16,  J<sub>out</sub> = 16, K = 1, R<sub>in</sub> = 395, R<sub>out</sub> = **395**

------

#### Layer_12: Concatenation | Output = 14×14×832 

R<sub>out</sub> [max] = **427**

---

#### Layer_13: Max Pool 3×3 | Stride = 2 | Output = 7×7×832

J<sub>in</sub> = 16,  J<sub>out</sub> = 32, K = 3, R<sub>in</sub> = 427, R<sub>out</sub> = **459**

---

#### Layer_14: Inception 5A

##### (Grouped Layers)

#### Layer_141: Convolution 1×1 | Stride = 1 | Output = 7×7×256

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 1, R<sub>in</sub> = 459, R<sub>out</sub> = **459**

------

#### Layer_142: Convolution 1×1 | Stride = 1 | Output = 7×7×160

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 1, R<sub>in</sub> = 459, R<sub>out</sub> = **459**

#### Layer_143: Convolution 3×3 | Stride = 1 | Output = 7×7×320

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 3, R<sub>in</sub> = 459, R<sub>out</sub> = **523**

------

#### Layer_144: Convolution 1×1 | Stride = 1 | Output = 7×7×32

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 1, R<sub>in</sub> = 459, R<sub>out</sub> = **459**

#### Layer_145: Convolution 5×5 | Stride = 1 | Output = 7×7×128

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 5, R<sub>in</sub> = 459, R<sub>out</sub> = **587**

------

#### Layer_146: Max Pool 3×3 | Stride = 1 | Output = 7×7×832 

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 3, R<sub>in</sub> = 459, R<sub>out</sub> = **523**

#### Layer_147: Convolution 1×1 | Stride = 1 | Output = 7×7×128 

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 1, R<sub>in</sub> = 523, R<sub>out</sub> = **523**

------

#### Layer_14: Concatenation | Output = 7×7×832 

R<sub>out</sub> [max] = **587**

---

#### Layer_15: Inception 5B

##### (Grouped Layers)

#### Layer_151: Convolution 1×1 | Stride = 1 | Output = 7×7×384

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 1, R<sub>in</sub> = 587, R<sub>out</sub> = **587**

------

#### Layer_152: Convolution 1×1 | Stride = 1 | Output = 7×7×192

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 1, R<sub>in</sub> = 587, R<sub>out</sub> = **587**

#### Layer_153: Convolution 3×3 | Stride = 1 | Output = 7×7×384

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 3, R<sub>in</sub> = 587, R<sub>out</sub> = **651**

------

#### Layer_154: Convolution 1×1 | Stride = 1 | Output = 7×7×48

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 1, R<sub>in</sub> = 587, R<sub>out</sub> = **587**

#### Layer_155: Convolution 5×5 | Stride = 1 | Output = 7×7×128

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 5, R<sub>in</sub> = 587, R<sub>out</sub> = **715**

------

#### Layer_156: Max Pool 3×3 | Stride = 1 | Output = 7×7×832 

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 3, R<sub>in</sub> = 587, R<sub>out</sub> = **651**

#### Layer_157: Convolution 1×1 | Stride = 1 | Output = 7×7×128 

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 1, R<sub>in</sub> = 651, R<sub>out</sub> = **651**

------

#### Layer_15: Concatenation | Output = 7×7×1024 

R<sub>out</sub> [max] = **715**

---

#### Layer_16: Average Pool 7×7 | Stride = 1 | Output = 1×1×1024 

J<sub>in</sub> = 32,  J<sub>out</sub> = 32, K = 7, R<sub>in</sub> = 715, R<sub>out</sub> = **907**

---

#### Layer_17: Dropout(0.4) | Output = 1×1×1024  

---

#### Layer_18: Dense(F.C.) | Output = 1×1×1000  or (,1000)

---

#### Layer_19: Softmax | Output = 1×1×1000  or (,1000)

---

## The Global Receptive Field of the network is *907*

---

