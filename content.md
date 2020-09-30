Creating AI(Deep Learning) solutions for business ? What You need to know.

Having a good deep learning model that solve a real problem can take a lot of
time and resources. There is a lot of materials in the internet about AI, but
from my observation the focus is still more in area of science not business.
Here I want to focus more on production related problems which maybe of interest
equally for business owners and IT developers.

The first papers about deep learning were published by Alexey Ivakhnenko[link]
in 1967 but only recently we are witnessing extreme growth in AI development.
Computing power available at the time of most fundamental discoveries in AI
was not enough to put development further.

The deep learning computational complexity problem was addressed with usage
of Graphical Processing Unit(GPU[link]) for Deep Learning algorithms. GPU is in
general a specialized computer processor responsible for rendering graphics for
display on an electronic device.

The game changer was introduced by 2012, when GPUs had evolved allowing on very
efficient parallel computing. Thanks to GPU, big and time consuming problems
can be know spitted into multiple problems and be solved at the same time.

As a business owner who wants to have AI solution You should be specially
interested about cloud server that will be used to train deep learning model.
Depending on algorithms used to train deep learning model and volume of data, a
professional developer will give You approximate time needed to train the
model.

What is CUDA ? CUDA is the parallel computing platform enable the computing 
power of GPUs for general purpose processing. CUDA was created by NIVIDIA[link]
which leads the development and production of highly efficient GPU processors.

https://developer.nvidia.com/cuda-zone
"CUDA is a parallel computing platform and programming model that makes using a
 GPU for general purpose computing simple and elegant. The developer still
 programs in the familiar C, C++, Fortran, or an ever expanding list of
 supported languages, and incorporates extensions of these languages in the
 form of a few basic keywords.
"

So what really drives AI development at the moment is hardware. The current
deep learning algorithms continue to solve more complex problems and the demand
on extra computational powers is still growing.

We have a GPU which is hardware, CUDA which is a platform with libraries that 
know how tu utilize GPU power and Deep Learning frameworks. Deep Learning
frameworks allows developer to focus on solving the problem, all needed
algorithms are already implemented, what is even more important those frameworks
are already built to be highly efficient by using GPU and CUDA.


https://developer.nvidia.com/deep-learning
""
Developing AI applications start with training deep neural networks with large
datasets. GPU-accelerated deep learning frameworks offer flexibility to design
and train custom deep neural networks and provide interfaces to commonly-used
programming languages such as Python and C/C++. Every major deep learning
framework such as TensorFlow, PyTorch, and others, are already GPU-accelerated,
so data scientists and researchers can get productive in minutes without any
GPU programming. 
"

PyTorch[link] is a deep learning framework I used so far. It is great because of
its pythonic structure and simplicity.

PyTorch is integrated with NIVIDIA deep learning library cuDNN.

https://developer.nvidia.com/cudnn
"The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated 
library of primitives for deep neural networks. cuDNN provides highly tuned 
implementations for standard routines such as forward and backward convolution,
pooling, normalization, and activation layers." 

Did You know that CUDA is not only for Deep Learning ? As developer You may
gain incredible performance boost in Your existing code by using CUDA. If You
work with Python, check Numba [https://numba.pydata.org/], it is developed by
Anaconda developers and supported among others by NIVIDIA.

If You are developer and want to know more about CUDA, check this article
https://developer.nvidia.com/blog/even-easier-introduction-cuda/.

PyTorch is supporting ONNX[https://onnx.ai/]. ONNX is a general format how
to represent deep learning model. So You could train Your deep learning model
on high performance cloud Amazon Web Servers and download it and try predicting
on Your local PC.

How to release AI solution for business and satisfy Your client ? First and the
most important inform client about costs, if it is too much for client try to
find existing solution, there is many pre-trained models that You could use.
If Your client will be specially interested about performance I would first 
start looking after solution proposed by hardware manufacturers like NIVIDIA
as they will for sure know best how to use hardware with deep learning to
achieve best results.

#aibusiness #ai #deeplearning #cuda #gpu #nividia #pytorch #tutorial