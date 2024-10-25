import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_results(results):
    """Visualize the results for accuracy and confusion matrix, and save the images to ../reports/images."""
    
    # Ensure the target folder exists
    output_folder = '../reports/images'
    os.makedirs(output_folder, exist_ok=True)
    
    # Visualization 1: Accuracy Bar Plot
    accuracies = [result['accuracy'] for result in results.values()]
    model_names = list(results.keys())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    # Save the plot
    plt.savefig(os.path.join(output_folder, 'model_accuracy_comparison.png'))
    plt.close()  # Close the plot to avoid display conflicts

    # Visualization 2: Confusion Matrices
    num_models = len(results)
    num_rows = (num_models + 1) // 2
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, num_rows * 6))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix for {name}')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    # Remove any extra subplots if there are fewer models than axes
    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])

    # Save the confusion matrix plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrices.png'))
    plt.close()