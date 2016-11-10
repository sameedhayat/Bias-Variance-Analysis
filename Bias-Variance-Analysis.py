import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

poly_degree = 4

# A helper function to make sure that we fit and evaluate matrices of
# correct sizes.
def verify_shapes(X=None, y=None):
    if X is not None:
        assert X.ndim == 2 and X.shape[1] == 1
    if y is not None:
        assert y.ndim == 2 and y.shape[1] == 1
        if X is not None:
            assert y.shape[0] == X.shape[0]


# The ground-truth that we want to approximate.
def target(X):
    verify_shapes(X)
    return np.sin(np.pi * X)


# Fit a constant model h(x) = b.
def fit_constant(X, y):
    verify_shapes(X, y)
    
    # Alternative: return y.mean(keepdims=True)
    return y.mean().reshape((-1, 1))
    

# Evaluate a constant model defined by its bias b.
def eval_constant(X, b):
    verify_shapes(X)
    
    # The result should have the same number of samples as X
    return np.repeat(b, X.shape[0], axis=0)

# Adding a bias "feature" to the data
def add_bias(X):
    bias_column = np.ones((X.shape[0], 1)) 
    X_aug = np.hstack([bias_column, X])
    return X_aug

# Fitting a linear model h(x) = wx + b.
def fit_linear(X, y):
    verify_shapes(X, y)
    X_aug = add_bias(X)
    inverse = np.linalg.pinv(X_aug.T @ X_aug)
    return inverse @ X_aug.T @ y

# Evaluating a linear model defined by its weights w.
def eval_linear(X, w):
    verify_shapes(X)
    X_aug = add_bias(X)
    return X_aug @ w

from sklearn.preprocessing import PolynomialFeatures
add_polynomials = PolynomialFeatures(degree=poly_degree).fit_transform

# Fitting a polynomial model of degree 'poly_degree'.
def fit_polynomial(X, y):
    verify_shapes(X, y)
    X_aug = add_polynomials(X)
    inverse = np.linalg.pinv(X_aug.T @ X_aug)
    return inverse @ X_aug.T @ y

# Evaluating a polynomial model defined by its weights w.
def eval_polynomial(X, w):
    verify_shapes(X)
    X_aug = add_polynomials(X)
    return X_aug @ w

n_datasets = 100
n_samples = 4

X_all = np.zeros((n_datasets, n_samples, 1))
y_all = np.zeros((n_datasets, n_samples, 1))

constant_models = np.zeros((n_datasets, 1, 1))
linear_models = np.zeros((n_datasets, 2, 1))
polynomial_models = np.zeros((n_datasets, poly_degree + 1, 1))

# Sampling the datasets uniformly for x in [-1, 1] and fitting the models
for i in range(n_datasets):
    X_all[i] = np.random.uniform(-1, 1, size=(n_samples, 1))
    y_all[i] = target(X_all[i])
    constant_models[i] = fit_constant(X_all[i], y_all[i])
    linear_models[i] = fit_linear(X_all[i], y_all[i])
    polynomial_models[i] = fit_polynomial(X_all[i], y_all[i])

# Determining the average model for each class by
# averaging over the first axis
average_constant_model = np.mean(constant_models, axis=0)
average_linear_model = np.mean(linear_models, axis=0)
average_polynomial_model = np.mean(polynomial_models, axis=0)

X_eval = np.linspace(-1, 1, 200).reshape((-1, 1))
y_eval = target(X_eval)

# Calculating the bias and variance for each class
def bias_variance_decomposition(models, average_model, eval_fct):
    model_evals = eval_fct(X_eval, models.T)
    average_eval = eval_fct(X_eval, average_model)
    bias = np.mean((average_eval - y_eval)**2)
    variance = np.mean((model_evals - average_eval)**2)

    return bias, variance

# Calculating the values is now just a function call for each model class
bias_constant_models, variance_constant_models = \
    bias_variance_decomposition(constant_models,
                                average_constant_model,
                                eval_constant)

bias_linear_models, variance_linear_models = \
    bias_variance_decomposition(linear_models,
                                average_linear_model,
                                eval_linear)
    
bias_polynomial_models, variance_polynomial_models = \
    bias_variance_decomposition(polynomial_models,
                                average_polynomial_model,
                                eval_polynomial)
    
# Printing the decomposition
bias_info = '''Bias:
    Constant models: {:.3f}
    Linear models: {:.3f}
    Polynomial models: {:.3f}'''
print(bias_info.format(float(bias_constant_models),
                                       float(bias_linear_models),
                                       float(bias_polynomial_models)))

variance_info = '''Variance:
    Constant models: {:.3f}
    Linear models: {:.3f}
    Polynomial models: {:.3f}'''
print(variance_info.format(float(variance_constant_models),
                                               float(variance_linear_models),
                                               float(variance_polynomial_models)))

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(7, 8))
axes[0].set_title('Constant models')
axes[1].set_title('Linear models')
axes[2].set_title('Polynomial models')

# Plotting all the fitted models
for dataset in range(n_datasets):
    constant = constant_models[dataset].flatten()
    axes[0].plot((-1, 1), (constant, constant), c='g', alpha=0.1)
    
    weights = linear_models[dataset]
    axes[1].plot((-1, 1), (weights[0] - weights[1], weights[0] + weights[1]),
                 c='g', alpha=0.1)
    y_poly = eval_polynomial(X_eval, polynomial_models[dataset])
    if y_poly is not None:
        axes[2].plot(X_eval, y_poly, c='g', alpha=0.1)
    
# Plotting the average models
if average_linear_model is not None:
    axes[1].plot((-1, 1), 
        (average_linear_model[0] - average_linear_model[1],
        average_linear_model[0] + average_linear_model[1]),
        c='r', lw=1.3, label='Average model')
if average_constant_model is not None:
    axes[0].plot((-1, 1), 
        (average_constant_model.flatten(), average_constant_model.flatten()),
        c='r', lw=1.3, label='Average model')
if average_polynomial_model is not None:
    axes[2].plot(X_eval, eval_polynomial(X_eval, average_polynomial_model), 
        c='r', lw=1.3, label='Average model')

for a in axes:
    a.plot(X_eval, target(X_eval), label='Target function')
    a.set_xlim((-1.1, 1.1))
    a.set_ylim((-1.5, 1.5))
fig.tight_layout()

#Cross-validation and Regularization
n_folds = 3

X_cv = X_all[:n_folds]
y_cv = y_all[:n_folds]

X_test = np.array([-0.97, -0.4, 0.7, 0.97]).reshape((-1, 1))
y_test = target(X_test)

# Calculating the root mean square error
def rmse(y, y_predicted):
    return np.sqrt(np.mean(2 * (y - y_predicted)**2))

def fit_ridge_regression(X, y, regularization_weight):
    verify_shapes(X, y)
    X_aug = add_polynomials(X)
    regularisation_term = regularization_weight * np.eye(X_aug.shape[1])
    inverse = np.linalg.pinv(regularisation_term + X_aug.T @ X_aug)
    return inverse @ X_aug.T @ y

training_errors = np.zeros((n_folds))
validation_errors = np.zeros((n_folds))
test_errors = np.zeros((n_folds))
models = np.zeros((n_folds, poly_degree + 1, 1))

# Here we define our cross-validation function specialized for
# our ridge regression models. The regularization weight is
# the only hyperparameter.
def cross_validate(regularization_weight):
    
    # Loop over the validation folds
    for i in range(n_folds):
        
        # Split the data into training and validation set
        # You can use the mask to select the correct training folds
        mask = np.arange(n_folds) != i
        X_train = X_cv[mask].reshape((-1, 1))
        y_train = y_cv[mask].reshape((-1, 1))
        verify_shapes(X_train, y_train)
        X_valid = X_cv[i]
        y_valid = y_cv[i]
        verify_shapes(X_valid, y_valid)
    
        # Train a model and determine the training and validation error
        models[i] = fit_ridge_regression(X_train, y_train,
                                         regularization_weight)
        
        # Calculating the training, validation and test error
       
        training_errors[i] = \
            rmse(y_train, eval_polynomial(X_train, models[i]))
        validation_errors[i] = \
            rmse(y_valid, eval_polynomial(X_valid, models[i]))
        test_errors[i] = \
            rmse(y_test, eval_polynomial(X_test, models[i]))
            
    # Calculating the statistics for the whole procedure
    avg_training_error = np.mean(training_errors)
    cross_validation_error = np.mean(validation_errors)
    avg_test_error = np.mean(test_errors)

    return avg_training_error, cross_validation_error, avg_test_error
    
# Set the strength of the regularisation here
regularization_weight = 0
avg_training_error, cross_validation_error, avg_test_error = \
    cross_validate(regularization_weight)

print('Regularization weight: {:.2e}'.format(regularization_weight))
print('Average training error: {:.3f}'.format(avg_training_error))
print('Cross-validation error: {:.3f}'.format(cross_validation_error))
print('Average test error: {:.3f}'.format(avg_test_error))

# Plotting: For 3 folds, we can still look at each model an how
fix, axes = plt.subplots(n_folds, 1, figsize=(12, 16))
for i, axis in enumerate(axes):
    axis.set_title('Fold {:d}'.format(i))
    axis.plot(X_eval, target(X_eval), label='Target function', alpha=0.5)
    if not np.any(np.isnan(models[i])):
        axis.plot(X_eval, eval_polynomial(X_eval, models[i]),
                  label='Fitted model', c='m')
    mask = np.arange(n_folds) != i
    axis.scatter(X_cv[mask].flatten(), y_cv[mask].flatten(),
                 label='Training samples', s=30, c='k')
    axis.scatter(X_cv[i].flatten(), y_cv[i].flatten(),
                 label='Validation samples', s=40, c='g')
    axis.scatter(X_test.flatten(), y_test.flatten(),
                 label='Test samples', s=50, c='r')
    axis.set_xlim((-1.1, 1.1))
    axis.set_ylim((-1.5, 1.5))
    axis.legend(loc='best')
plt.show(block=True)
