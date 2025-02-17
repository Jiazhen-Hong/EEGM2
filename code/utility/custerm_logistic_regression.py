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




# def make_representation(full_model, loader, device="cuda", target_layer_name="bottleneck"):
#     """
#     Extract representations from a specific layer of the model or the final output.

#     Args:
#         full_model (torch.nn.Module): The pre-trained full model.
#         loader (DataLoader): DataLoader for the dataset.
#         device (str): The device to use ('cuda' or 'cpu').
#         target_layer_name (str): The name of the specific layer to extract representations from.
#                                  Use "final" to extract features from the final output.

#     Returns:
#         torch.Tensor: Feature representations from the target layer or final output.
#         torch.Tensor: Corresponding labels.
#     """
#     # Check if the model is wrapped in DataParallel
#     if isinstance(full_model, torch.nn.DataParallel):
#         model = full_model.module  # Access the actual model
#     else:
#         model = full_model

#     if target_layer_name == "final":
#         # Directly extract final outputs
#         features = []
#         labels = []

#         # Ensure model is in evaluation mode
#         full_model.eval()

#         with torch.no_grad():
#             for inputs, targets in loader:
#                 inputs = inputs.to(device)
#                 outputs = full_model(inputs)  # Forward pass through the entire model
#                 features.append(outputs.cpu())  # Append the final outputs as features
#                 labels.append(targets)

#         return torch.cat(features), torch.cat(labels)

#     else:
#         # Extract features from a specific intermediate layer
#         target_layer = getattr(model, target_layer_name, None)
#         if target_layer is None:
#             raise ValueError(f"Layer '{target_layer_name}' not found in the model.")

#         intermediate_features = []

#         # Define a hook function to capture the target layer's output
#         def hook_fn(module, input, output):
#             intermediate_features.append(output)

#         # Register the hook to the target layer
#         hook_handle = target_layer.register_forward_hook(hook_fn)

#         features = []
#         labels = []

#         # Ensure model is in evaluation mode
#         full_model.eval()

#         with torch.no_grad():
#             for inputs, targets in loader:
#                 inputs = inputs.to(device)
#                 intermediate_features.clear()  # Clear previous intermediate features
#                 full_model(inputs)  # Forward pass through the entire model
#                 if not intermediate_features:
#                     raise RuntimeError(f"No features captured from layer '{target_layer_name}'.")
#                 features.append(intermediate_features[0].cpu())  # Append captured features
#                 labels.append(targets)

#         # Remove the hook to avoid memory leaks
#         hook_handle.remove()

#         return torch.cat(features), torch.cat(labels)
    

def find_layer_by_name(model, layer_name):
    """
    Recursive function to find a layer within a nested module structure.

    Args:
        model (torch.nn.Module): The model to search.
        layer_name (str): The name of the target layer.

    Returns:
        torch.nn.Module: The target layer if found.

    Raises:
        ValueError: If the layer is not found.
    """
    components = layer_name.split(".")
    current_layer = model
    for comp in components:
        if hasattr(current_layer, comp):
            current_layer = getattr(current_layer, comp)
        else:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")
    return current_layer


def make_representation(full_model, loader, device="cuda", target_layer_name="bottleneck"):
    """
    Extract representations from a specific layer of the model using a forward hook.

    Args:
        full_model (torch.nn.Module): The pre-trained full model.
        loader (DataLoader): DataLoader for the dataset.
        device (str): The device to use ('cuda' or 'cpu').
        target_layer_name (str): The name of the specific layer to extract representations from.

    Returns:
        torch.Tensor: Feature representations from the target layer.
        torch.Tensor: Corresponding labels.
    """
    # Check if the model is wrapped in DataParallel
    if isinstance(full_model, torch.nn.DataParallel):
        model = full_model.module  # Access the actual model
    else:
        model = full_model

    # Find the target layer (supports nested layers)
    target_layer = find_layer_by_name(model, target_layer_name)

    intermediate_features = []

    # Define a hook function to capture the target layer's output
    def hook_fn(module, input, output):
        intermediate_features.append(output)

    # Register the hook to the target layer
    hook_handle = target_layer.register_forward_hook(hook_fn)

    features = []
    labels = []

    # Ensure model is in evaluation mode
    full_model.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            intermediate_features.clear()  # Clear previous intermediate features
            full_model(inputs)  # Forward pass through the entire model
            if not intermediate_features:
                raise RuntimeError(f"No features captured from layer '{target_layer_name}'.")
            features.append(intermediate_features[0].cpu())  # Append captured features
            labels.append(targets)

    # Remove the hook to avoid memory leaks
    hook_handle.remove()

    return torch.cat(features), torch.cat(labels)