import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_data(data_list,file_name, title, printing_y, printing_x, type):
    plt.figure(figsize=(10,6))
    legend_properties = {'weight':'bold'}
    if (type == "loss"):
        plt.plot(data_list, label='Value loss', marker='.',markersize=8, linestyle='-', linewidth=2, color="red")
    else:
        plt.plot(data_list, label='Value accuracy', marker='.',markersize=8, linestyle='-',linewidth=2, color="blue")
    
    # Adding labels and title
    plt.tick_params(axis='x', labelsize=14)  # Asse x
    plt.tick_params(axis='y', labelsize=14) 
    plt.xlabel(f"{printing_x}", fontsize=17, weight='bold')
    
    plt.ylabel(f'Value {printing_y}', fontsize=17, weight='bold')
    plt.title(f"{title}", fontsize=20,  weight='bold')
    
    plt.legend(fontsize=14, prop=legend_properties)
    plt.grid(color="gray",linestyle="solid")
    

    # Save the plot as a PNG file
    plt.savefig(file_name, format='pdf', dpi=300, bbox_inches='tight')  # Save with high resolution
    print("Plot saved")
    
def plot_data_double(train_acc, val_acc ,file_name, title, printing_y, printing_x, type):
    plt.figure(figsize=(10,6))
    legend_properties = {'weight':'bold'}
   
    plt.plot(train_acc, label='Train accuracy', marker='.',markersize=8, linestyle='-', linewidth=2, color="red")
    
    plt.plot(val_acc, label='Validation accuracy', marker='.',markersize=8, linestyle='-',linewidth=2, color="blue")
    
    
    
    # Adding labels and title
    plt.tick_params(axis='x', labelsize=14)  # Asse x
    plt.tick_params(axis='y', labelsize=14) 
    plt.xlabel(f"{printing_x}", fontsize=17, weight='bold')
    
    plt.ylabel(f'Value {printing_y}', fontsize=17, weight='bold')
    plt.title(f"{title}", fontsize=20,  weight='bold')
    
    plt.legend(fontsize=14, prop=legend_properties)
    plt.grid(color="gray",linestyle="solid")
    

    # Save the plot as a PNG file
    plt.savefig(file_name, format='pdf', dpi=300, bbox_inches='tight')  # Save with high resolution
    print("Plot saved")
    
def plot_data_double_loss(train_acc, val_acc ,file_name, title, printing_y, printing_x, type):
    plt.figure(figsize=(10,6))
    legend_properties = {'weight':'bold'}
   
    plt.plot(train_acc, label='Train loss', marker='.',markersize=8, linestyle='-', linewidth=2, color="red")
    
    plt.plot(val_acc, label='Validation loss', marker='.',markersize=8, linestyle='-',linewidth=2, color="blue")
    
    
    
    # Adding labels and title
    plt.tick_params(axis='x', labelsize=14)  # Asse x
    plt.tick_params(axis='y', labelsize=14) 
    plt.xlabel(f"{printing_x}", fontsize=17, weight='bold')
    
    plt.ylabel(f'Value {printing_y}', fontsize=17, weight='bold')
    plt.title(f"{title}", fontsize=20,  weight='bold')
    
    plt.legend(fontsize=14, prop=legend_properties)
    plt.grid(color="gray",linestyle="solid")
    

    # Save the plot as a PNG file
    plt.savefig(file_name, format='pdf', dpi=300, bbox_inches='tight')  # Save with high resolution
    print("Plot saved")

def compute_confusion_matrix(all_true_labels,predicted_labels,path, title):
    cm = confusion_matrix(all_true_labels, predicted_labels, normalize="true")
    #cm[0,0] += 0.01
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',annot_kws={"fontsize": 14}, xticklabels=['Neutral', 'Happy', 'Angry', 'Sad'], yticklabels=['Neutral', 'Happy', 'Angry', 'Sad'])
    plt.tick_params(axis='x', labelsize=16)  # Asse x
    plt.tick_params(axis='y', labelsize=16) 
    plt.xlabel('Predicted Labels', fontsize=15)
    plt.ylabel('True Labels', fontsize=15)
    plt.title(title, fontsize=18, weight='bold')
    plt.savefig(path, format='pdf', dpi=300)
    plt.close()