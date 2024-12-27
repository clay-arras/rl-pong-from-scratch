question: 
- how is gradient descent implemented here? what are the gradients for each of the weights?

# derivative of RELU is 0 if x < 0 else 1
# derivative of softmax is s_i(1-s_i) if i = j else -s(i)s(j)