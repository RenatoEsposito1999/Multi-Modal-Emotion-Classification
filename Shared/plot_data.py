import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_data(data_list,file_name, title, printing_y, printing_x):
    plt.figure(figsize=(10,6))
    plt.plot(data_list, label='Value', marker='o', linestyle='-', color='b')

    # Adding labels and title
    plt.xlabel(f"{printing_x}", fontsize=12)
    plt.ylabel(f'Value {printing_y}', fontsize=12)
    plt.title(f"{title}", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot as a PNG file
    plt.savefig(file_name, dpi=300, bbox_inches='tight')  # Save with high resolution
    print("Plot saved")

def compute_confusion_matrix(all_true_labels,predicted_labels,path):
    cm = confusion_matrix(all_true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral', 'Happy', 'Angry', 'Sad'], yticklabels=['Neutral', 'Happy', 'Angry', 'Sad'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(path, format='jpeg')
    plt.close()