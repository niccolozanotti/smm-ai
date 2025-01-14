# HW 2: SVD and PCA for Machine Learning

## Visualizing dyad
Consider an image from `skimage.data`. For simplicity, say that $X \in \mathbb{R}^{m \times n}$ is the matrix representing that image. You are asked to visualize the dyad of the SVD Decomposition of $X$ and the result of compressing the image via SVD. In particular:

* Load the image into memory and compute its SVD;
* Visualize some of the dyad $\sigma_i u_i v_i^T$ of this decomposition. What do you notice?
* Plot the singular values of $X$. Do you note something?
* Visualize the $k$-rank approximation of $X$ for different values of $k$. What do you observe?
* Compute and plot the approximation error $|| X − X_k ||_F$ for increasing values of $k$, where $X_k$ is the $k$-rank approximation of $k$.
* Plot the compression factor: $c_k = 1 − \frac{k(m+n+1)}{mn}$ for increasing values of $k$.
* Compute the value $k$ such that $c_k = 0$ (i.e. when the compressed image requires the same amount of informations of those of the uncompressed image). What is the approximation error for this value of $k$? Comment.

It is strongly recommended (but not mandatory) to consider a grey-scale image for this exercise. You can also use an image downloaded from the web. Clearly, if your image will be an RGB image, then its shape will be `(m, n, 3)`, where the last dimension corresponds to the three channels (Red, Green, and Blue). Every point discussed in the Homework has to be done on each channel separately, and then aggregated back to an RGB image.

## Classification of MNIST Digits with SVD Decomposition.
For this exercise we aim to develop a classification algorithm on MNIST digits using SVD decomposition.
We recall that, given a matrix $X \in \mathbb{R}^{d \times N}$ and its SVD decomposition $X = USV^T$, it is easy to show that an orthogonal basis for the space of the columns is given by the first $p$ columns of the matrix $U$, where $p = rank(X)$ is equal to the number of non-zero singular values of $X$. We will make use of the space of the columns defined by the $U$ matrix and the following Theorem:

**Theorem 1.** Let $W$ be a subspace of $\mathbb{R}^d$ with $dim W = s$, and let ${w_1, \dots, w_s}$ be an orthogonal basis of $W$. Then, for any $x \in \mathbb{R}^d$, the projection $x^\perp$ of $x$ onto $W$ has the following form:

$$
x^\perp = \frac{x \cdot w_1}{w_1 \cdot w_1} w_1 + \dots + \frac{x \cdot w_s}{w_s \cdot w_s} w_s.
$$

**Corollary 1.1.** Let $X \in \mathbb{R}^{d \times N}$ be a matrix with SVD decomposition $X = USV^T$, since $p = rank(X)$ is the dimension of the space defined by the columns of $X$ and the columns of $U$, ${u_1, \dots, u_p}$ are an orthonormal basis for that space, the projection of an $d$-dimensional vector $x$ on this space can be easily
computed as:

$$
x^\perp = U(U^T x).
$$

Consider as an example a binary classification problem, where we want to distinguish between hand-written digit representing numbers 3 and 4. We will refer to the class of the images representing number 3 as $C_1$, and to the set of images representing the number 4 as $C_2$. Let $N_1$ be the number of elements in $C_1$, and $N_2$ be the number of elements in $C_2$. Let $X_1 \in \mathbb{R}^{d \times N_1}$ be the matrix such that its columns are a flatten version of each digit in $C_1$, $X_2 \in \mathbb{R}^{d \times N_2}$ be the matrix such that its columns are a flatten version of each digit in $C_2$, and consider:

$$
X_1 = U_1S_1V_1^T, \\
X_2 = U_2S_2V_2^T,
$$

the SVD decomposition of the two matrices.

If $x \in \mathbb{R}^{d}$ is a new, unknown digit, we can predict its class through our classification algorithm by projecting it onto the spaces induced by the SVD of $X_1$ and $X_2$ via:

$$
x_1^\perp = U_1(U_1^T x), \\
x_2^\perp = U_2(U_2^T x),
$$

and classify $x$ as an element of either $C_1$ or $C_2$ based on $||x − x_1^\perp ||_2$ being greater of lower than $||x−x_2^\perp ||_2$, respectively. In this exercise, you are required to implement this idea in Python.

> The description provided up to this point is only meant to understand the basic idea of the algorithm we aim to implement. From now on, I will list the point you are effectively required to implement in Python, therefore I will start re-defining some quantities, possibly overlapping with some discussion already made.


1. Implement the binary classification algorithm discussed above for the digits 3 and 4 of MNIST dataset. Follow these steps:
   * Download the MNIST dataset from [kaggle.com/datasets/animatronbot/mnist-digit-recognizer](https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer) and load it into memory by following the steps we did in the [PCA class](https://devangelista2.github.io/statistical-mathematical-methods/ML/PCA.html). When loaded into memory, this dataset appear as an array with shape $42000 \times 785$ , containining the flattened version of $42000$ $28 \times 28$ grayscale handwritten digits, plus a column representing the true class of the corresponding digit. By pre-processing the data as we did in class, you should obtain a matrix `X` containing the flattenened digits, with shape `(784, 42000)`, and a vector `Y` of the associated digit value, with a shape of `(42000,)`.
   * Write a function taking as input an index value `idx` and visualizes the image of `X` in the corresponding index (i.e. `X[idx, :]`). Use the function `plt.imshow`.
   * Filter from `X` only those elements that corresponds to digits 3 or 4. This can be done, for example, by using the boolean slicing of `numpy` arrays, as already discussed in class.
   * Split the obtained dataset in training and testing in a proportion of $80 : 20$. From now on, we will only consider the training set. The test set will be only used at the end of the exercise to test the algorithm.
   * Call `X1` and `X2` the submatrices of the training set, filtered by the two selected digits, corresponding to those element associated with number 3 (class `C1`), and with number 4 (class `C2`).
   * Compute the SVD decomposition of `X1` and `X2` with `np.linalg.svd(matrix, full_matrices=False)` and denote the $U$-part of the two decompositions as `U1` and `U2`.
   * Take an unknown digit $x$ from the test set, and compute $x_1^\perp = U_1(U_1^T x)$ and $x_2^\perp = U_2(U_2^T x)$.
   * Compute the distances $d_1 = || x − x_1^\perp ||_2$ and $d_2 = || x − x_2^\perp||_2$, and classify $x$ as $C_1$ if $d_1 < d_2$, as $C_2$ if $d_2 < d_1$.
   * Repeat the experiment for different values of $x$ in the test set. Compute the misclassification rate for this algorithm.
   * Repeat the experiment for different digits other than 3 or 4. There is a relationship between the visual similarity of the digits and the classification error?
   * Comment the obtained results.


> Given a classification algorithm $f(x)$, which maps an input image $x$ into its predicted class, the misclassification rate on the test set is defined as:
> $$
 MR = \frac{1}{N_{test}} \sum_{i=1}^{N_test} \iota(f(x_i) == y_i),
> $$
> where $N_{test}$ is the number of elements in the test set, $(x_i, y_i)$ represents the $i$-th element of the test set, while $\iota(f(x_i) == y_i)$ is a function which is equal to 0 if $f(x_i)$ is equal to the true class $y_i$, while it is equal to 1 if $f(x_i)$ guesses the wrong digit (i.e. it is different from $y_i$). More simply, the Misclassification Rate represent the average number of error of the model over the test set.

2. The extension of this idea to the multiple classification task is trivial. Indeed, if we have more than 2 classes (say, $k$ different classes) $C_1, \dots, C_k$, we just need to repeat the same procedure as before for each matrix $X_1, \dots, X_k$ to obtain the distances $d_1, \dots, d_k$. Then, the new digit $x$ from the test set will be classified as $C_i$ if $d_i$ is lower that $d_j$ for each $j = 1,...,k$. Repeat the exercise above with a 3-digit example. Comment the differences.

## Clustering with PCA
In this exercise we want to analyse the ability of PCA in clustering data by projecting very high-dimensional datapoints to 2 or 3 dimensions. In particular, consider the same MNIST dataset used in the previous exercise. You are asked to:
* Load and pre-process the dataset as did in the previous exercise, to get the matrix `X` with shape `(784, 42000)`, and the associated vector `Y`.
* Choose a number of digits (for example, 0, 6 and 9) and extract from `X` and `Y` the sub-dataset containing only the considered digits, as did in the previous exercise.
* Set $N_{train} < N$ and randomly sample a training set with $N_{train}$ datapoints from  `X` and `Y`. Call them `X_train` and `Y_train`. Everything else is the test set. Call them `X_test` and `Y_test`, correspondingly. This has to be done **after** filtering out the selected digits from `X` and `Y`.
* Implement the algorithms computing the PCA of `X_train` with a fixed value of $k$. Visualize the results (for $k = 2$) and the position of the centroid of each cluster. The clusters are identified by projecting `X_train` via PCA to its low-dimension version `Z_train`, and then splitting it into sets (say, `Z1`, `Z2`, `Z3`) based on the digit that was represented in that position before the PCA projection. Each set `Z1`, `Z2`, `Z3` represents a cluster, of which we can easily compute the centroid.
* Compute, for each cluster, the average distance from its centroid. Which property of PCA projection does this quantity measure?
* By keeping the **same** projection matrix `P` from the train set, project the test set `X_test` on the low-dimensional space.
* Consider the clusters in `X_test` by considering the informations on `Y_test`, similarly to what we did on the previous point. Consider the centroids computed from the training set. For each cluster in the test set, compute the average distance to the corresponding centroid (from the train set). Comment the results;
* Define a classification algorithm in this way: given a new observation `x`, compute the distance between `x` and each cluster centroid computed on the training set. Assign `x` to the class corresponding the the closer centroid. Compute the misclassification rate of this algorithm on the test set;
* Repeat this experiment for different values of $k$ and different digits. What do you observe?
* Compare this classification algorithm with the one defined in the previous exercise. Which performs better?