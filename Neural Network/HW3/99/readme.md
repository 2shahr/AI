An auto-encoder (AE) is a neural network that learns to copy its input to its output.
It has an internal (hidden) layer that describes a code used to represent the input, and
it is formed by two main parts: an encoder that maps the input into the code (latent
space), and a decoder that maps the code to a reconstruction of the original input.
One of the main usage of AEs is to reduce the dimension by extracting meaningful
features in the latent space (code layer). Representing data in a lower-dimensional
space can improve performance on different tasks, such as classification and
clustering.
