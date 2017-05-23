The goal of this project is to use machine learning to solve a problem that arises in Information Retrieval, one known as the Learning to rank (LeToR) problem. 

1. The project has a specific requirement in terms of dividing the Microsoft LETOR 4.0 data set and the synthetic data set into training, validation and testing set.
2. The data set is supposed to be partitioned into a training set which takes around 80% of the total, a validation set that comprises of 10% of the total and a testing set that takes the rest.
3. In view of this requirement, I have identified that there are 2 ways to partition this data set.
4. Partition can be done randomly in which any rows that constitute 80% of the total 69623 rows, which is 55601 query-document pairs (rows) will be taken as the training set, 10% of the remaining 20%, which is 7011 rows will be taken as the validation set and the other 7011 (10% of the total) will be taken as the testing set.
5. Another way to partition the data is to directly take the first 80% rows as training set, the next 10% rows as the validation set and the rest 10% as the testing set.
6. I have partitioned the data using the second way and hence have the following values for the training, validation and testing set:
   Training data – X and Y matrices that split the main data into 46 feature vectors and the resulting output matrix which contains relevance score. X and Y constitute 55601 rows, which is 80% of the data.
  Validation data – Xvalid and Yvalid matrices contain next 7011 rows each and have 46 feature vectors and corresponding relevance score.
  Testing data – Xtest and Ytest matrices contain the remaining 7011 rows each and have 46 feature vectors and the relevance score.

Hyper-parameter Tuning:

7.  Hyper parameters M, μj, Σj, λ, η(τ) need to be evaluated first in order to proceed further with the linear regression model training.
8. Basis function M can be chosen using grid search.
9. Alternatively, by assuming M as some integer, say 4 or 5, we can get different weight vectors and train linear regression model.
10. Later on, validation data can be trained on the obtained weight vectors and the hyper parameters will then be adjusted if the expected result is ambiguous.
11. It should be noted that the value of M should not be too small or too large.
12. For this project, I have chosen 4 as M value and trained the model parameter w on the training set initially.
13. Weights and regularized weights calculated using both closed form solution and stochastic gradient function are as follows:  Weight from closed form solution:

  [[ 0.01588176],
  [ 1.39858078],
  [-0.87881206],
  [-0.08195343]]
  
14. Weight from stochastic gradient function: [[ -2.85644991e-04], [ 9.99792989e-01], [ -2.25282447e-04], [ -2.32610177e-04]]
15. Here, for regularized weights and error calculation, λ value is needed.
16. λ can also be calculated using grid search. It should be in the interval of 0 and 1 and hence, a random floating point in the range (0,1) is taken as λ.
17. Using regularized function λ, the weights are calculated to be as follows:
18. Regularized weight using closed form solution: [[ 0.01696059], [ 1.39168956], [-0.87625263], [-0.07928387]]
19. Regularized weight using stochastic gradient descent function: [[ -2.85511518e-04], [ 9.99433444e-01], [ -2.25178493e-04], [ -2.32501222e-04]]  μj, Σj, η(τ) need to be calculated before the calculation of phi matrix too, the explanation of which is covered in the upcoming sections.
20. After obtaining weight vector, the validation data is used to validate the accuracy of the model trained using the hyper parameters assumed in the initial stage.
21. By taking different values of M, we obtain different weights and Root mean square errors.
22. When M is 4, the ERMS is of optimal value.

23. Mean and Sigma values are calculated directly from the data.

Eta:

24. Learning rate, η(τ) can be either fixed or variable.
25. The learning rate should be in the range of 0 and 1, hence I have assumed it to be 0.5 (fixed) in the initial calculation.
26. The weights that are obtained using this value of Eta are presented in the initial sections of this report.
27. During validation, I have employed multiple values of Eta such as 1, 0.7, 0.4.
28. It can be inferred that ERMS is optimal when Eta is 0.5 in all forms as when compared to other values of Eta.
