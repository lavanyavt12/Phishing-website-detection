import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, model_name):
    """Print evaluation metrics for a model"""
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def save_results(model_name, accuracy, classification_rep, confusion_mat, history=None):
    """Save model results to files with proper subfolder structure"""
    # Create base directories
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    metrics_dir = os.path.join(results_dir, 'metrics')
    plots_dir = os.path.join(results_dir, 'plots')
    conf_mat_dir = os.path.join(plots_dir, 'confusion_matrices')
    history_dir = os.path.join(plots_dir, 'training_history')
    
    # Create all directories
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(conf_mat_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to text file
    metrics_file = os.path.join(metrics_dir, f'{model_name}_metrics_{timestamp}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"{model_name} Model Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(classification_rep))
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(conf_mat_dir, f'{model_name}_confusion_matrix_{timestamp}.png'))
    plt.close()
    
    # Save training history for neural networks
    if history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(history_dir, f'{model_name}_training_history_{timestamp}.png'))
        plt.close()
        
    print(f"Results saved for {model_name} in appropriate subfolders under {results_dir}")