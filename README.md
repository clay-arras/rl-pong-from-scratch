question: 
- how is gradient descent implemented here? what are the gradients for each of the weights?
- how does softmax gradient work? what are the shapes of these vectors?

# derivative of RELU is 0 if x < 0 else 1
# derivative of softmax is s_i(1-s_i) if i = j else -s(i)s(j)

<!-- the above gives you the Jacobian matrix. To get the gradient for a specfic weight, I'm assuming you need to just sum up along an axis -->
<!-- resolution: there is no softmax gradient ... instead there are two hidden layers -->

