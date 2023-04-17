This is an axial plane of a knee magnetic resonance image. 

We have a neural network that aims to classify MR images to determine whether they have a specific kind of abnormality.

The input grayscale image is actually a matrix that can be reshaped into a 1D vector to fit the input neurons.

The neural network functions as expected due to feed-forward and back-propagation. During back-propagation, the network can be trained by optimizing each parameter through gradient descent.

The gradient descent algorithm decreases each parameter by a tiny learning rate (alpha) multiplied by the gradient of the loss function with respect to that parameter. 

However, if we modify the input X by adding the gradient of the loss function with respect to each pixel of X multiplied by a tiny attack rate (epsilon), X will ascend in the direction of the anti-gradient.

To simplify the computation, we can substitute the gradient with its sign.

As a result, the neural network will misclassify the updated version of input X. This is how the Fast Gradient Sign Method works.

Although FGSM can cause the network to misclassify, if we feed the network that the altered input X has the same label as the original input X, the neural network can learn from the perturbation.