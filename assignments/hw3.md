# HW 3: Optimization

## Optimization via Gradient Descent
In this Homework, we will consider a general optimization problem:

$$
x^* = \arg\min_{x \in \mathbb{R}^n}f(x).
$$

where, $f: \mathbb{R}^n \to \mathbb{R}$ is a differentiable function for which we know how to compute $\nabla f(x)$.
This is done by the Gradient Descent (GD) method: an iterative algorithm that, given an initial iterate $x_0 \in \mathbb{R}^n$ and a positive parameter $\alpha_k > $ called *step size*, computes:

$$
x_{k+1} = x_k − \alpha_k \nabla f (x_k).
$$

You are asked to implement the GD method in Python and to test it with some exemplar functions. In particular:

*  Write a script that implement the GD algorithm with fixed step size (i.e. no backtracking), with the input-output structure discussed in the first Exercise of the Gradient Descent section (https://devangelista2.github.io/statistical-mathematical-methods/Optimization/GD.html).
* Write a script that implement the GD algorithm with backtracking, with the input-output structure discussed in the second Exercise of the Gradient Descent section (https://devangelista2.github.io/statistical-mathematical-methods/Optimization/GD.html).
* Test the algorithm above on the following functions:
    1. $f: \mathbb{R}^2 \to \mathbb{R}$ such that:

       $$
       f(x_1, x_2) = (x_1 - 3)^2 + (x_2 - 1)^2, 
       $$

       for which the true solution is $x^* = (3, 1)^T$.

    2. $f: \mathbb{R}^2 \to \mathbb{R}$ such that:

       $$
       f(x_1, x_2) = 10(x_1 − 1)^2 + (x_2 − 2)^2, 
       $$

       for which the true solution is $x^* = (1, 2)^T$.
    3. $f: \mathbb{R}^n \to \mathbb{R}$ such that:

       $$
       f(x) = \frac{1}{2}|| Ax - b ||_2^2, 
       $$

       where $A \in \mathbb{R}^{n times n}$ is the Vandermonde matrix associated with the vector $v \in \mathbb{R}^n$ that contains $n$ equispaced values in the interval $[0,1]$, and $b \in \mathbb{R}^n$ is computed by first setting $x^* = (1, 1, \dots ,1)^T$, and then $b = A x^*$. Try for different values of $n$ (e.g. $n = 5,10,15, \dots$).
    4. $f: \mathbb{R}^n \to \mathbb{R}$ such that:

       $$
       f(x) = \frac{1}{2} || Ax - b ||_2^2 + \frac{\lambda}{2} ||x||_2^2, 
       $$

       where $A \in \mathbb{R}^{n times n}$ and $b \in \mathbb{R}^n$ are the same of the exercise above, while $\lambda$ is a fixed value in the interval $[0, 1]$. Try different values of $\lambda$ and comment the result.
    5. $f: \mathbb{R} \to \mathbb{R}$ such that:

       $$
       f(x) = x^4 + x^3 - 2x^2 - 2x.
       $$

* For each of the functions above, test the GD method with and without backtracking, trying different values for the step size $\alpha > 0$ when backtracking is not employed. Comment on the results.
* Plot the value of $||\nabla f(x_k)||_2$ as a function of $k$, check that it goes to zero, and compare the convergence speed (in terms of the number of iterations $k$) for the different values of $\alpha > 0$ and with backtracking.
* For each of the points above, use:
  - `x0` = $(0, 0, \dots, 0)^T$ (except for function 5, which is discussed in the following point),
  - `kmax` = 100,
  - `tolf` = `tolx` = `1e-5`. 
  Also, when the true solution $x^*$ is given, plot the error $||x_k−x^*||_2$ as a function of $k$.
* Plot the graph of the non-convex function 5 in the interval $[−3,3]$, and test the convergence of GD with different values of `x0` (of your choice) and different step-sizes. When is the convergence point the global minimum?
* *Hard (optional):* For functions 1 and 2, show the contour plot around the true minimum and visualize the path described by the iterations, i.e. representing on the contour plot the position of each iterate computed by the GD algorithm. See the `plt.contour` documentation.

## Optimization via Stochastic Gradient Descent
Consider a dataset $(X,Y)$, where:

$$
X = \begin{bmatrix} x^1 & x^2 & \dots & x^N \end{bmatrix} \in \mathbb{R}^{d \times N}, \qquad Y = \begin{bmatrix} y^1 & y^2 & \dots & y^N \end{bmatrix} \in \mathbb{R}^N,
$$

together with a model $f_\theta(x)$, with vector of parameters $\theta$. **Training** a ML model requires solving:

$$
\theta^* = \arg\min_{\theta} \ell(\theta; X, Y) = \arg\min_{\theta} \sum_{i=1}^N \ell_i(\theta; x^i, y^i). 
$$

Since the optimization problem above is written as a sum of independent terms that only depends on the single datapoints, it satisfies the hypothesis for the application of the Stochastic Gradient Descent (SGD) algorithm, which articulates as follows:

* Given an integer `batch_size`, *randomly* extract a sub-dataset $\mathcal{M}$ such that $|\mathcal{M}| = `batch_size`$ from the original dataset. Note that the random sampling at each iteration has to be done without replacement.
* Compute the gradient of the loss function on the sampled batch $\mathcal{M}$ as:

  $$
  \nabla \ell(\theta; \mathcal{M}) = \frac{1}{| \mathcal{M} |} \sum_{i \in \mathcal{M}} \nabla \ell (\theta; x^i, y^i),
  $$
* Compute one single iteration of the GD algorithm on the direction described by $\nabla \ell(\theta; \mathcal{M})$:

  $$
  \theta_{k+1} = \theta_k - \alpha_k \nabla \ell(\theta_k; \mathcal{M}),
  $$

* Repeat until the full dataset has been extracted. When this happens, we say that we completed an **epoch** of the SGD method. Repeat this procedure for a number of epochs equal to a parameter `n_epochs`, given as input.

Consider the dataset `poly_regression_large.csv`, provided on Virtuale, and let $f_\theta(x)$ be a polynomial regression model, as discussed in https://devangelista2.github.io/statistical-mathematical-methods/regression_classification/regression.html.

* Split the dataset into training and test set as in the Homework 2, with a proportion of 80% training and 20% test. 
* Fix a degree $K$ for the polynomial. 
* Train the polynomial regression model on the training set via the Stochastic Gradient Descent algorithm.
* Train the polynomial regression model on the training set via the Gradient Descent algorithm.
* Train the polynomial regression model on the `poly_regression_small.csv` dataset. Use the full dataset for this test, without splitting it into training and test set.
* Compare the performance of the three regression model computed above. In particular, if $(X_{test}, Y_{test})$ is the test set from the `poly_regression_large.csv` dataset, for each of the model, compute:

  $$
  Err = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} (f_\theta(x^i) - y^i)^2,
  $$

  where $N_{test}$ is the number of elements in the test set, $(x^i, y^i)$ are the input and output elements in the test set. Comment the performance of the three models.

* Repeat the experiment by varying the degree $K$ of the polynomial. Comment the results.
* Set $K=5$ (so that the polynomial regression model is a polynomial of degree 4). Compare the parameters learned by the three models with the true parameter $\theta^* = [0, 0, 4, 0, -3]$.