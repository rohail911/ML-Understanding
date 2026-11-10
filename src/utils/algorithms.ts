import { AlgorithmMetadata } from '../types/ml';

export const ALGORITHMS: AlgorithmMetadata[] = [
  {
    id: 'kmeans',
    name: 'K-Means Clustering',
    category: 'Unsupervised',
    tagline: 'Centroid-driven partitioning of unlabelled dots into harmonious clusters',
    description:
      'K-Means partitions data dots into K distinct groups. Centroid dots are initialized, each data dot assigns itself to the closest centroid, and centroids dynamically migrate to the center of their dot family until convergence.',
    mathFormula: 'J = \\sum_{i=1}^{k} \\sum_{x \\in S_i} ||x - \\mu_i||^2',
    formulaExplanation:
      'Minimizes the Within-Cluster Sum of Squares (WCSS). Centroids μ_i pull nearby data points x to minimize total dispersion.',
    primaryMetric: 'Inertia (WCSS)',
    color: '#06b6d4', // cyan-500
    accentRgb: '6, 182, 212',
    iconName: 'Network',
    keyConcepts: ['Centroid Migration', 'Voronoi Tessellation', 'Convergence', 'K-Means++ Seeding'],
  },
  {
    id: 'knn',
    name: 'K-Nearest Neighbors',
    category: 'Supervised',
    tagline: 'Instance-based classification decided by local democratic voting',
    description:
      'KNN does not learn an explicit model. Instead, every new query dot measures its Euclidean distance to all known dots, selects the K closest neighbors, and lets them vote on the predicted category.',
    mathFormula: 'P(y = c | x) = \\frac{1}{K} \\sum_{i \\in N_K(x)} I(y_i = c)',
    formulaExplanation:
      'The class probability is the proportion of the K nearest neighbor dots belonging to class c, optionally weighted by inverse distance 1/d.',
    primaryMetric: 'Neighbor Consensus',
    color: '#f59e0b', // amber-500
    accentRgb: '245, 158, 11',
    iconName: 'Radar',
    keyConcepts: ['Local Neighborhoods', 'Distance Metrics', 'Boundary Smoothness', 'Lazy Learning'],
  },
  {
    id: 'linear-regression',
    name: 'Linear Regression & Gradient Descent',
    category: 'Supervised',
    tagline: 'Continuous line fitting guided by the gradient down a loss valley',
    description:
      'Models continuous relationships between feature coordinates. Watch the parameter dot descend step-by-step into the parabolic loss valley (Mean Squared Error bowl), iteratively aligning the line of best fit.',
    mathFormula: '\\theta_j := \\theta_j - \\alpha \\frac{\\partial}{\\partial \\theta_j} J(\\theta)',
    formulaExplanation:
      'Parameters θ move opposite to the gradient vector scaled by the learning rate α, progressively minimizing the squared residuals.',
    primaryMetric: 'R² Score / MSE',
    color: '#3b82f6', // blue-500
    accentRgb: '59, 130, 246',
    iconName: 'TrendingUp',
    keyConcepts: ['Residual Error', 'Loss Landscape', 'Learning Rate α', 'Ordinary Least Squares'],
  },
  {
    id: 'logistic-regression',
    name: 'Logistic Regression',
    category: 'Supervised',
    tagline: 'Calibrated class probabilities through smooth sigmoid boundaries',
    description:
      'Unlike linear regression, logistic regression maps continuous linear combinations through the sigmoid function σ(z) to output true probabilities between 0 and 1, creating smooth transition gradients across dot territories.',
    mathFormula: 'P(y=1 \\mid x) = \\sigma(w^T x + b) = \\frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2 + b)}}',
    formulaExplanation:
      'Evaluates log-odds of membership. The decision threshold (default 0.5) divides the space into predicted classes while conveying prediction confidence.',
    primaryMetric: 'Binary Cross-Entropy Loss',
    color: '#6366f1', // indigo-500
    accentRgb: '99, 102, 241',
    iconName: 'Activity',
    keyConcepts: ['Sigmoid Function', 'Probability Surface', 'Log-Odds', 'Cross-Entropy'],
  },
  {
    id: 'svm',
    name: 'Support Vector Machine',
    category: 'Supervised',
    tagline: 'Finding the widest safety margin between opposing dot classes',
    description:
      'SVM constructs the optimal separating hyperplane that maximizes the geometric distance to the closest data dots (the Support Vectors). In non-linear mode, dots are lifted into higher dimensions via the Kernel Trick.',
    mathFormula: '\\min_{w, b} \\frac{1}{2} ||w||^2 + C \\sum_{i} \\xi_i',
    formulaExplanation:
      'Maximizes margin width 2/||w|| while penalizing classification errors ξ_i balanced by the regularization budget C.',
    primaryMetric: 'Geometric Margin',
    color: '#8b5cf6', // violet-500
    accentRgb: '139, 92, 246',
    iconName: 'Shield',
    keyConcepts: ['Hyperplane', 'Support Vectors', 'Soft Margin (C)', 'Kernel Elevation'],
  },
  {
    id: 'decision-tree',
    name: 'Decision Tree Partitioning',
    category: 'Supervised',
    tagline: 'Hierarchical recursive slicing of the dot universe into pure regions',
    description:
      'A decision tree recursively splits space using perpendicular thresholds on X or Y coordinates. At each split, it selects the boundary that maximizes information gain or minimizes Gini impurity among dots.',
    mathFormula: 'Gini = 1 - \\sum_{i=1}^{C} p_i^2',
    formulaExplanation:
      'Measures the probability of misclassifying a randomly chosen dot in a partition. Splits strive for 0 impurity (100% pure clusters).',
    primaryMetric: 'Tree Purity / Depth',
    color: '#f97316', // orange-500
    accentRgb: '249, 115, 22',
    iconName: 'GitFork',
    keyConcepts: ['Recursive Partitioning', 'Gini Impurity', 'Entropy', 'Overfitting Depth'],
  },
  {
    id: 'random-forest',
    name: 'Random Forest Ensemble',
    category: 'Supervised',
    tagline: 'Aggregating bootstrap tree voters to eliminate variance and overfitting',
    description:
      'A single decision tree easily overfits complex dot patterns. Random Forest trains an ensemble of randomized trees on bootstrap dot subsets and averages their votes, smoothing sharp step edges into robust probabilistic contours.',
    mathFormula: '\\hat{y} = \\frac{1}{B} \\sum_{b=1}^{B} T_b(x)',
    formulaExplanation:
      'Bagging (Bootstrap Aggregation) combines B decorrelated decision trees, dramatically reducing variance without increasing bias.',
    primaryMetric: 'Ensemble Consensus',
    color: '#10b981', // emerald-500
    accentRgb: '16, 185, 129',
    iconName: 'Trees',
    keyConcepts: ['Bagging', 'Ensemble Voting', 'Variance Reduction', 'Feature Subspacing'],
  },
  {
    id: 'naive-bayes',
    name: 'Gaussian Naive Bayes',
    category: 'Supervised',
    tagline: 'Bayesian posterior contours assuming conditionally independent feature distributions',
    description:
      'Fits 2D Gaussian bell curves (mean μ, variance σ²) around each dot class. By assuming features X₁ and X₂ are conditionally independent, it calculates the posterior probability P(Class | X) with minimal computational cost.',
    mathFormula: 'P(C_k \\mid x) \\propto P(C_k) \\cdot \\mathcal{N}(x_1; \\mu_{k1}, \\sigma_{k1}^2) \\cdot \\mathcal{N}(x_2; \\mu_{k2}, \\sigma_{k2}^2)',
    formulaExplanation:
      'Applies Bayes Theorem with the naive independence assumption. Generates smooth elliptical likelihood contours around class centers.',
    primaryMetric: 'Posterior Likelihood',
    color: '#eab308', // yellow-500
    accentRgb: '234, 179, 8',
    iconName: 'Binary',
    keyConcepts: ['Bayes Theorem', 'Prior vs Posterior', 'Likelihood Ellipses', 'Feature Independence'],
  },
  {
    id: 'neural-net',
    name: 'Neural Network / MLP',
    category: 'Deep Learning',
    tagline: 'Non-linear feature synthesis through layers of firing neuron dots',
    description:
      'Data dots pass forward through input synapses and non-linear hidden activation neurons, firing forward to compute output probability contours and backpropagating error gradients.',
    mathFormula: 'a^{(l)} = \\sigma(W^{(l)} a^{(l-1)} + b^{(l)})',
    formulaExplanation:
      'Activations propagate layer by layer through linear combinations passed through non-linear activation functions (ReLU, Sigmoid, Tanh).',
    primaryMetric: 'Binary Cross-Entropy Loss',
    color: '#ec4899', // pink-500
    accentRgb: '236, 72, 153',
    iconName: 'Cpu',
    keyConcepts: ['Forward Activation', 'Backpropagation', 'Activation Curves', 'Decision Contours'],
  },
  {
    id: 'dbscan',
    name: 'DBSCAN Density Clustering',
    category: 'Unsupervised',
    tagline: 'Discovering arbitrary-shaped clusters and rejecting outlier noise dots',
    description:
      'Unlike K-Means which requires knowing K upfront and assumes spherical clusters, DBSCAN groups dots based on local density. Dots with ≥ MinPts within radius ε are Core dots; connected cores form clusters of any shape, while isolated dots become Noise!',
    mathFormula: 'N_\\epsilon(p) = \\{q \\in D \\mid ||p - q|| \\le \\epsilon\\}',
    formulaExplanation:
      'Expands clusters through density-reachability: a dot p is a Core dot if |N_ε(p)| ≥ MinPts. Automatically discards sparse outliers.',
    primaryMetric: 'Core / Noise Ratio',
    color: '#14b8a6', // teal-500
    accentRgb: '20, 184, 166',
    iconName: 'CircleDot',
    keyConcepts: ['Epsilon Radius ε', 'MinPts Threshold', 'Core vs Border Dots', 'Outlier Filtering'],
  },
  {
    id: 'hierarchical',
    name: 'Hierarchical Clustering',
    category: 'Unsupervised',
    tagline: 'Agglomerative bottom-up tree merging of closest dot pairs',
    description:
      'Starts with every dot in its own solitary cluster. At each iteration, the two closest clusters are merged based on linkage criteria (single, complete, or average distance) building an intuitive dendrogram tree.',
    mathFormula: 'D(A, B) = \\min / \\max / \\text{avg} \\{ d(x, y) \\mid x \\in A, y \\in B \\}',
    formulaExplanation:
      'Iteratively merges the pair of clusters with minimum pairwise linkage distance until the desired number of clusters is reached.',
    primaryMetric: 'Dendrogram Cut Level',
    color: '#f43f5e', // rose-500
    accentRgb: '244, 63, 94',
    iconName: 'GitMerge',
    keyConcepts: ['Agglomerative Merging', 'Dendrogram Tree', 'Single/Complete Linkage', 'Distance Matrix'],
  },
  {
    id: 'pca',
    name: 'Principal Component Analysis',
    category: 'Dimensionality Reduction',
    tagline: 'Discovering the axis of maximum variance and collapsing dimensions',
    description:
      'PCA rotates the coordinate frame along the primary direction where dots vary the most. High-dimensional dots project perpendicularly onto the first principal component vector, compressing information while retaining maximum variance.',
    mathFormula: '\\Sigma v = \\lambda v \\quad \\text{where } \\Sigma = \\frac{1}{n} X^T X',
    formulaExplanation:
      'Eigenvectors v of covariance matrix Σ indicate the principal axes, and eigenvalues λ measure the variance preserved along each axis.',
    primaryMetric: 'Variance Retained (%)',
    color: '#10b981', // emerald-500
    accentRgb: '16, 185, 129',
    iconName: 'Compass',
    keyConcepts: ['Covariance Matrix', 'Eigenvectors', 'Orthogonal Projection', 'Dimensionality Reduction'],
  },
];
