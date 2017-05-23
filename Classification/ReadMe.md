This project aims to implement Classification algorithm to recognize a 28 * 28 grayscale handwritten digit image and identify it as a digit among 0,1,2,..,9.

1. Logistic Regression, single hidden layer neural networks and convolutional neural networks algorithms are developed so as to train them on the MNIST dataset and USPS dataset.
2. The MNIST dataset is downloaded from the internet and is loaded into Python as a 2-Dimensional array using cPicle() library.
3. The data is divided into training, validation and test sets.
4. The weights are obtained using training data and accuracy of these weights is tested on validation and testing data.

Logistic Regression:
5. 1-of-K coding scheme has been used here.
6. Initial weights are randomly generated in the confinements of [784,10].
7. From these weights, the activation vector a, is calculated and the resulting y(x), which is the logistic regression model is obtained using the exponential probability. The gradient of the errors function is calculated accordingly. 
8. Initially, when the t vector is taken from the training data, it will be a [50000,1] array.
9. This needs to be converted to [50000,10] array so as to make this equation work.
10. It can be done using one hot encoding where the target value in each row of the t vector is taken, assigned a value 1 in the corresponding index of the tj vector while all the other values in the row will be 0.
11. Using the gradient of the error function and some random eta value (between 0 and 1), the new weights are calculated and updated.
12. This process is run for 100-500 times and the resulting y array is taken and compared to t vector from the MNIST data.
13.  Based on the number of equalities obtained, accuracy of the algorithm is calculated and the maximum obtained is 91.28 for training data, 90.12 for validation and 89.05 for testing data for eta = 0.5.

Single Layer Neural Network:
14. A single layer neural network with one hidden layer is used here.
15. The weights of the hidden units wji and wkj are initialized randomly between 0 and 1.
16. Wji vector will have dimensions [784,M] for the training data of MNIST dataset where in M is the number of nodes in the hidden layer.
17. Wkj vector will have dimensions [M,10] where M is the number of nodes in hidden layer and 10 is the classification digit image range (there are 10 digits here).
18. For each row, we calculate zj, ak and yk vectors where zj is the activation of the hidden layer and h(.) function is the activation function of the hidden layer.
19. Ideally, h(.) can be either logistic sigmoid, hyperbolic tangent or rectified linear unit.
20. Here, we have considered h(.) function to be a logistic sigmoid.
21. Cross entropy function is used to find the error and the gradient of the error function is derived from this using derivation w.r.t both the weights wji and wkj.
22. After obtaining the gradients, stochastic gradient descent can be used to train the neural network.
23. The new weights are calculated by subtracting the original weights with (eta * gradient function).
24. After running this process for all the rows in the data for n number of iterations, we get the final y vector which will be compared with the t vector taken from the training data (the original classified data).
25. Both these vectors are compared the resulting accuracy of the algorithm will be known.
26. When 500 hidden nodes are taken in a single layer and eta is 0.01, the highest accuracy found for the training data is , validation data is 92.51%, for testing data is 92.00%.
27. In this algorithm, there are two hyper parameters that we tune so as to get desired results, namely eta and number of nodes.
