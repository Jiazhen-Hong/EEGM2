import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_log(event_path, tag_name):
    """
    Load a specific scalar tag from a TensorBoard event file.
    
    Args:
        event_path (str): Path to the TensorBoard event file (e.g., event.out.tfevents).
        tag_name (str): Name of the tag to load (e.g., "Loss/train", "Accuracy/val").
    
    Returns:
        steps (list): List of steps.
        values (list): List of scalar values for the given tag.
    """
    # Load the TensorBoard log
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()  # Load the logs

    # Get the scalar data for the specified tag
    steps = []
    values = []
    if tag_name in event_acc.Tags()['scalars']:
        for scalar_event in event_acc.Scalars(tag_name):
            steps.append(scalar_event.step)
            values.append(scalar_event.value)
    else:
        raise ValueError(f"Tag '{tag_name}' not found in the event file!")
    
    return steps, values

def calculate_average(values):
    """
    Calculate the average of a list of values.
    
    Args:
        values (list of float): List of scalar values.
    
    Returns:
        float: Average of the list.
    """
    return sum(values) / len(values) if values else 0.0

def plot_tensorboard_logs(event_files, tag_name, labels=None):
    """
    Plot multiple TensorBoard logs for a specific tag and calculate average values.
    
    Args:
        event_files (list of str): List of TensorBoard event files to load.
        tag_name (str): Name of the tag to plot (e.g., "Loss/train").
        labels (list of str): Labels for each plot (same length as event_files).
    """
    plt.figure(figsize=(10, 8))

    colors = ['red', 'g', 'b', 'c', 'm', 'y', 'k', '#ff5733', '#33ff57', '#3357ff']

    for idx, event_file in enumerate(event_files):
        steps, values = load_tensorboard_log(event_file, tag_name)
        avg_value = calculate_average(values)  # Calculate average
        print(f"{labels[idx] if labels else f'Run {idx+1}'} (Avg: {avg_value:.4f})")
        color = colors[idx % len(colors)]
        label = labels[idx] if labels else f"Run {idx+1}"
        plt.plot(steps, values, label=label, linewidth=3, color=color, alpha=0.5)

    plt.xlabel("Epochs", fontsize=20, fontweight='bold')
    plt.ylabel("Time (s)", fontsize=20, fontweight='bold')

    # Set tick parameters to make them bold
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=10, width=2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("timel.png", dpi=300)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Path to your TensorBoard event files
    event_files = [
        "./TUAB100event/EEGM2.tfevents",
        "./TUAB100event/EEGM2-S1.tfevents",
        #"./TUAB100event/EEGM2-S2.tfevents",
        #"./TUAB100event/EEGM2-S3.tfevents",
        #"./TUAB100event/EEGM2-S4.tfevents"
    ]

    # Labels for each plot
    #labels = ["EEGM2", "EEGM2-S1", "EEGM2-S2", "EEGM2-S3", "EEGM2-S4"]
    #labels = ["EEGM2", "EEGM2-S1"]
    labels = ["EEGM2", "EEGM2-S1"]

    # Specify the scalar tag you want to plot
    tag_name = "Time/epoch_duration"  # Replace with the tag in your logs

    # Plot the logs
    plot_tensorboard_logs(event_files, tag_name, labels=labels)