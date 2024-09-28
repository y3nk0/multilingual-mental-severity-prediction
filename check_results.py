import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from retry import retry
import time
from sklearn.metrics import f1_score
# import plotly.express as px
from sklearn.metrics import classification_report
# import plotly.io as pio
from depression_utils import depression_count_answers_general
from suicide_utils import suicide_count_answers_general
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
# from erroranalysis import ErrorAnalyzer, ErrorHeatMap

# language = 'french'
# language = 'greek'
# method = 'add_shot_'
# method = ''
model = 'gpt-3.5-turbo'
# model = ''
model = 'gpt-4o-mini'
dataset = 'depression'
languages = ['english', 'turkish', 'french', 'portuguese', 'german', 'greek', 'finnish']

if dataset=='depression':
    df = pd.read_csv("data/Depression_Severity_Dataset-main/Reddit_depression_dataset.csv", quotechar='"')
    class_labels = ['Minimum', 'Mild', 'Moderate', 'Severe']
    severity_mapping = {
        'minimum': 0,
        'mild': 1,
        'moderate': 2,
        'severe': 3
    }
    # Apply the mapping to the 'Severity' column
    df['label'] = df['label'].map(severity_mapping)

elif dataset=='suicide':
    df = pd.read_csv("data/suicide/Labelled_tweets.tsv", header=0, delimiter="\t", quoting=3)
    df = df.rename(columns={'tweet': 'text', 'category': 'label'})
    # Define class labels (replace with your actual class labels)
    class_labels = ['Class 0', 'Class 1', 'Class 2']

# Organize F1-scores into lists for each class
f1_scores_per_class = {label: [] for label in class_labels}

f = open('results/'+method+model+'_all_'+dataset+'.txt', 'w')
for language in languages:
    print(language)
    if dataset=='suicide':
        f_trans = open('results/suic_preds_'+language+'.txt', 'r')
        lines = f_trans.readlines()
        f_trans.close()
        # preds = count_greek_answers(lines)
        preds = suicide_count_answers_general(lines, language)
    elif dataset=='depression':
        f_trans = open('results/'+method+model+'_reddit_preds_'+language+'.txt', 'r')
        lines = f_trans.readlines()
        f_trans.close()
        # preds = count_greek_answers(lines)
        preds = depression_count_answers_general(lines, language)

    # import pdb; pdb.set_trace()
    accuracy = accuracy_score(df['label'], preds)
    print(accuracy)

    report = classification_report(df['label'], preds)
    f.write(language+"\nAccuracy: "+str(accuracy)+"\nReport\n: "+str(report)+"\n\n")

    f1 = f1_score(df['label'], preds, average='macro')
    f.write(f"Macro F1 Score: {f1}\n")
    f1 = f1_score(df['label'], preds, average='micro')
    f.write(f"Micro F1 Score: {f1}\n\n")

    # Compute F1-score for each class
    f1_scores = f1_score(df['label'], preds, average=None)
    print(f1_scores)
    for label, score in zip(class_labels, f1_scores):
        f1_scores_per_class[label].append(score)

    # Generate the confusion matrix
    cm = confusion_matrix(df['label'], preds)

    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print("Per-class accuracy:", per_class_accuracy)
    f.write("Per-class accuracy:" + str(per_class_accuracy)+"\n")

    # Normalize the confusion matrix by row (true class)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Percentage)')
    # plt.show()
    # Save the plot as a high-quality PDF file with a tight layout
    plt.tight_layout()
    plt.savefig('figures/'+method+model+'_confusion_matrix_percentage_'+language+'.pdf', format='pdf', dpi=300)
    # Close the plot to free up memory
    plt.close()
    # Initialize ErrorAnalyzer with only labels and predictions
    # analyzer = ErrorAnalyzer(df['label'], preds)
    #
    # # Create an error heatmap
    # heatmap = ErrorHeatMap(analyzer)
    # heatmap.visualize()

    # # Prepare data for Plotly
    # data = {
    #     'Class': [f'Class {class_label}' for class_label, _ in f1_scores],
    #     'F1-score': [score for _, score in f1_scores]
    # }
    # # Create a boxplot using Plotly
    # fig = px.box(data, y='F1-score', x='Class', title='Boxplot of F1-scores per Class', points="all")
    # pio.write_image(fig, 'figures/f1_scores_per_class.pdf', format='pdf')

f.close()

# Create a boxplot for each class
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=[f1_scores_per_class[label] for label in class_labels])
# plt.xticks(ticks=range(len(class_labels)), labels=class_labels)
# plt.xlabel('Class')
# plt.ylabel('F1-Score')
# plt.title('Boxplot of F1-Scores per Class')
# # plt.show()
# plt.tight_layout()
# plt.savefig('figures/'+dataset+'_boxplot.pdf', format='pdf', dpi=300)
# # Close the plot to free up memory
# plt.close()

# Create a boxplot for each class with enhanced visuals
# plt.figure(figsize=(14, 10))
# plt.rcParams['xtick.major.pad']='2'
# # sns.set(style="whitegrid")
#
# # Use a vibrant color palette for better visuals
# palette = sns.color_palette("Set3")[4:8]
#
# # Plot the data
# sns.boxplot(data=[f1_scores_per_class[label] for label in class_labels], palette=palette)
#
# # Customize the plot with larger font sizes and new colors
# plt.xticks(ticks=range(len(class_labels)), labels=class_labels, fontsize=28)
# plt.yticks(fontsize=28)
# plt.xlabel('Class', fontsize=28)
# plt.ylabel('F1-Score', fontsize=28)
# plt.title('Boxplot of F1-Scores per Class', fontsize=28)
# # Add y-axis grid
# plt.grid(axis='y')
#
# # Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig('figures/' + method + '_' + model + '_' + dataset + '_boxplot.pdf', format='pdf', dpi=300)
#
# # Close the plot to free up memory
# plt.close()
