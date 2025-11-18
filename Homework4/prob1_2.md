================================================================================
PROBLEM 1_a: SVMs and PCA
================================================================================

Loading data...
Training set: 6000 samples, 5000 features

--------------------------------------------------------------------------------
Performing PCA...
Data shape: (6000, 5000)
M (number of data points): 6000
n (number of features): 5000
Covariance matrix shape: (5000, 5000)
Computing eigenvalues and eigenvectors...

--------------------------------------------------------------------------------
Part 1: Top 6 EIGENVALUES OF THE COVARIANCE MATRIX:
--------------------------------------------------------------------------------
Eigenvalue 1: 16440913.57
Eigenvalue 2: 12009807.65
Eigenvalue 3: 9726589.20
Eigenvalue 4: 7483337.87
Eigenvalue 5: 6784373.74
Eigenvalue 6: 5485503.60

Total variance: 321657360.76
Variance explained by top 6: 57930525.637875
Percentage explained by top 6: 18.01%

--------------------------------------------------------------------------------
Part 2: Building set K - VARIANCE THRESHOLDS
--------------------------------------------------------------------------------

k99 (explains 99.0% variance):
  k = 2695 components

k95 (explains 95.0% variance):
  k = 1761 components

k90 (explains 90.0% variance):
  k = 1320 components

k80 (explains 80.0% variance):
  k = 855 components

k75 (explains 75.0% variance):
  k = 700 components

Set K = {'k99': 2695, 'k95': 1761, 'k90': 1320, 'k80': 855, 'k75': 700}

--------------------------------------------------------------------------------
PART 3: PROJECTING DATA AND TRAINING SVM
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
Training SVM with k99 = 2695 components
--------------------------------------------------------------------------------
Projected training data shape: (6000, 2695)
Tuning hyperparameters on validation set...
Best hyperparameters: C=1, sigma=10000
Validation accuracy: 97.80%
Test accuracy: 98.00%
Test error: 2.00%

--------------------------------------------------------------------------------
Training SVM with k95 = 1761 components
--------------------------------------------------------------------------------
Projected training data shape: (6000, 1761)
Tuning hyperparameters on validation set...
Best hyperparameters: C=1, sigma=10000
Validation accuracy: 97.80%
Test accuracy: 98.00%
Test error: 2.00%

--------------------------------------------------------------------------------
Training SVM with k90 = 1320 components
--------------------------------------------------------------------------------
Projected training data shape: (6000, 1320)
Tuning hyperparameters on validation set...
Best hyperparameters: C=1, sigma=10000
Validation accuracy: 97.80%
Test accuracy: 98.00%
Test error: 2.00%

--------------------------------------------------------------------------------
Training SVM with k80 = 855 components
--------------------------------------------------------------------------------
Projected training data shape: (6000, 855)
Tuning hyperparameters on validation set...
Best hyperparameters: C=1, sigma=10000
Validation accuracy: 97.80%
Test accuracy: 98.20%
Test error: 1.80%

--------------------------------------------------------------------------------
Training SVM with k75 = 700 components
--------------------------------------------------------------------------------
Projected training data shape: (6000, 700)
Tuning hyperparameters on validation set...
Best hyperparameters: C=10, sigma=10000
Validation accuracy: 98.00%
Test accuracy: 97.80%
Test error: 2.20%

--------------------------------------------------------------------------------
RESULTS SUMMARY
--------------------------------------------------------------------------------
Model      k        C        sigma      Valid Acc    Test Acc     Test Error
--------------------------------------------------------------------------------
k99        2695     1.0      10000.00   97.80        98.00        2.00
k95        1761     1.0      10000.00   97.80        98.00        2.00
k90        1320     1.0      10000.00   97.80        98.00        2.00
k80        855      1.0      10000.00   97.80        98.20        1.80
k75        700      10.0     10000.00   98.00        97.80        2.20

--------------------------------------------------------------------------------
BEST PCA MODEL: k80
  k = 855
  C = 1
  sigma = 10000
  Test Error = 1.80%
--------------------------------------------------------------------------------

================================================================================
BASELINE: SVM WITHOUT PCA (All Original Features)
================================================================================

Training SVM with ALL 5000 original features (no PCA)
Tuning hyperparameters on validation set...
  Trying C=0.1, sigma=3000... Val Acc: 69.60%
  Trying C=0.1, sigma=10000... Val Acc: 92.80%
  Trying C=0.1, sigma=30000... Val Acc: 93.20%
  Trying C=1, sigma=3000... Val Acc: 69.60%
  Trying C=1, sigma=10000... Val Acc: 97.80%
  Trying C=1, sigma=30000... Val Acc: 97.20%
  Trying C=10, sigma=3000... Val Acc: 69.60%
  Trying C=10, sigma=10000... Val Acc: 97.80%
  Trying C=10, sigma=30000... Val Acc: 97.40%

Best hyperparameters: C=1, sigma=10000
Validation accuracy: 97.80%
Training final model with best hyperparameters...

================================================================================
BASELINE RESULTS
================================================================================
Features used: ALL 5000 original features (no PCA)
Best C: 1
Best sigma: 10000
Validation accuracy: 97.80%
Test accuracy: 98.00%
Test error: 2.00%
================================================================================

================================================================================
FINAL COMPARISON
================================================================================

Best PCA Model (k80 with 855 components):
  Test Error: 1.80%

Baseline Model (all 5000 features):
  Test Error: 2.00%

Difference: 0.20%
PCA model is BETTER by 0.20%
================================================================================
