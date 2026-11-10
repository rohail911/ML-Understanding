export type AlgorithmId =
  | 'kmeans'
  | 'knn'
  | 'linear-regression'
  | 'svm'
  | 'neural-net'
  | 'pca'
  | 'decision-tree'
  | 'dbscan'
  | 'logistic-regression'
  | 'naive-bayes'
  | 'random-forest'
  | 'hierarchical';

export type Category = 'Supervised' | 'Unsupervised' | 'Deep Learning' | 'Dimensionality Reduction';

export interface AlgorithmMetadata {
  id: AlgorithmId;
  name: string;
  category: Category;
  tagline: string;
  description: string;
  mathFormula: string;
  formulaExplanation: string;
  primaryMetric: string;
  color: string;
  accentRgb: string;
  iconName: string;
  keyConcepts: string[];
}

export interface DataDot {
  id: string;
  x: number; // Normalized 0 to 1
  y: number; // Normalized 0 to 1
  label?: number; // 0 or 1 for classification
  continuousTarget?: number; // for regression
  cluster?: number; // assigned cluster in kmeans/dbscan/hierarchical
  isSupportVector?: boolean;
  alpha?: number;
  residual?: number;
  highlighted?: boolean;
  dbscanType?: 'core' | 'border' | 'noise';
  dbscanCluster?: number;
  isNoise?: boolean;
  isCore?: boolean;
}

export interface Centroid {
  x: number;
  y: number;
  color: string;
  id: number;
}

export interface AlgorithmParams {
  // K-Means
  kmeansK: number;
  kmeansInit: 'kmeans++' | 'random';
  showVoronoi: boolean;

  // KNN
  knnK: number;
  knnWeighting: 'uniform' | 'distance';
  showDecisionBoundary: boolean;

  // Linear Regression & Gradient Descent
  learningRate: number;
  regIterations: number;
  fitIntercept: boolean;
  showResiduals: boolean;

  // SVM
  svmC: number;
  svmKernel: 'linear' | 'rbf';
  svmRbfGamma: number;

  // Neural Net
  nnHiddenNeurons: number;
  nnActivation: 'relu' | 'sigmoid' | 'tanh';
  nnEpochs: number;

  // PCA
  pcaComponents: number;
  showProjections: boolean;

  // Decision Tree
  treeMaxDepth: number;
  treeCriterion: 'gini' | 'entropy';

  // DBSCAN
  dbscanEps: number;
  dbscanMinPts: number;

  // Logistic Regression
  logRegLr: number;
  logRegThreshold: number;

  // Naive Bayes
  nbVarSmoothing: number;

  // Random Forest
  rfNumTrees: number;
  rfMaxDepth: number;

  // Hierarchical
  hierarchicalClusters: number;
  hierarchicalLinkage: 'single' | 'complete' | 'average';
}

export interface SimulationState {
  isRunning: boolean;
  step: number;
  speed: number; // 1 to 5
  history: any[];
}

export interface DotInspection {
  dot: DataDot;
  algorithmId: AlgorithmId;
  insights: {
    title: string;
    description: string;
    stats: { label: string; value: string }[];
  };
}
