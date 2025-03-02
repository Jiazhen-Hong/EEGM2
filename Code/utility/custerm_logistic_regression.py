from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import torch 

# FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. Use OneVsRestClassifier(LogisticRegression(..)) instead. Leave it to its default value to avoid this warning.
def fit_lr(features, y, seed=3407, MAX_SAMPLES=100000):
    """
    Train a logistic regression model on given features and labels.
    
    Args:
        features (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        MAX_SAMPLES (int): Maximum number of samples for training (default: 100000).
    
    Returns:
        sklearn.pipeline.Pipeline: Trained logistic regression model with standardization.
    """
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=seed, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                random_state=seed,
                max_iter=1000000
            )
        )
    )
    pipe.fit(features, y)
    return pipe

