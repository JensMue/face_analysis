from matplotlib import pyplot as plt

def model_performance(history, model_name = None):
    
    sns.set()
    fig = plt.figure(0, (12, 4))
    ax = plt.subplot(1, 2, 1)
    sns.lineplot(history.epoch, history.history['accuracy'], label='train')
    sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
    plt.title('Accuracy')
    plt.tight_layout()

    ax = plt.subplot(1, 2, 2)
    sns.lineplot(history.epoch, history.history['loss'], label='train')
    sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
    plt.title('Loss')
    plt.tight_layout()
    plt.show()
    plot = plt.savefig(model_name + '_history.png')
    return plot

# model_performance(baseline_history, model_name = "baseline")
import csv
import seaborn as sns
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report

emo_dict = {0:"neutral", 1:"happy", 2:"surprise", 3:"sad", 4:"anger", 5:"disgust", 6:"fear"}
expressions = ['neutral','happy', 'surprise', 'sad','anger', 'disgust','fear']

def evaluate(model, model_name, test_ds, batch_size):
    
    """Evaluates the model """
    
    test_ds = test_ds.batch(batch_size)
    
    labels = emo_dict.values()
    
    y_pred = model.predict(test_ds, verbose=0)
    
    label_true = np.concatenate([y for x, y in test_ds], axis=0)
    label_pred = np.argmax(y_pred, axis=1)
    
    bal_accuracy = sklearn.metrics.balanced_accuracy_score(label_true,
                                                          label_pred,
                                                          sample_weight=None,
                                                          adjusted=False)
    
    bal_acc_dict = {"Balanced Accuracy Score" : bal_accuracy}
    
    class_report = classification_report(label_true,
                                         label_pred,
                                         target_names = expressions,
                                         output_dict = True)
    
    class_report["Balanced Accuracy Score"] = bal_acc_dict #append to class report to export in one csv
    
    print(class_report)   
    
    w = csv.writer(open(model_name + "_metrics.csv", "w"))
    for key, val in class_report.items():
        w.writerow([key, val])
        
    conf = confusion_matrix(label_true, label_pred)
    conf = conf / np.max(conf)

    _, ax = plt.subplots(figsize=(8, 6))
    ax = sns.heatmap(conf, annot=True, cmap='YlGnBu', 
                     xticklabels=labels, 
                     yticklabels=labels)
    
    plot = plt.savefig(model_name + '_confmat.png')
    plt.show() 
    return class_report

# report = evaluate(baseline_model, model_name = "baseline")


def save_architecture(model, rel_path, model_as_str):
    
    """saves the model architecture relative to users home directory
    Inputs is the defined model architecture, the relative path,
    and the model name as str"""
    
    with open( os.path.join(os.path.expanduser('~'),
                rel_path, model_as_str + ".json"), "w") as json_file:
                json_file.write(model.to_json())


