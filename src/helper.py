import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

def clean_columns(column_list):
    """
    Function to convert a dataframe column names into lowercase and snake case
    
    INPUT
    :column_list: a list of column names, the datatypes inside the list are all string
    
    OUTPUT
    :final_list: a list of column names converted into lowercase and snake case
    """
    all_cols = column_list
    
    modified_list = []

    for item in all_cols:
        item = str(item).lower()
        modified_item = re.sub(r'[^a-zA-Z0-9]', '_', item)
        modified_list.append(modified_item)
    
    final_list = []
    
    for i in modified_list:
        cleaned_column_name = re.sub(r'_+', '_', i)
        final_list.append(cleaned_column_name)
    
    final_list = [col.strip('_') for col in final_list]
        
    return final_list

def save_file(name, obj):
    """
    Function to save an object as pickle file
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def load_file(name):
    """
    Function to load a pickle object
    """
    return pickle.load(open(name, "rb"))

def create_bar_chart(df,col,rotation=0):
    """
    Function to create a barchart
    
    INPUT
    :df: the pandas dataframe of interest
    :col: the column of interest, the datatype is string
    :rotation: how many degrees of rotation for the x-axis names if neccessary. Default is 0
    
    OUTPUT
    None. The function only shows the plot. 
    """
    value_counts_series = df[col].value_counts()

    # Create a bar chart using Seaborn
    sns.barplot(x=value_counts_series.index, y=value_counts_series.values)

    # Add labels and title
    plt.xlabel(col.replace('_',' ').title())
    plt.ylabel('Count')
    plt.title('Distribution of ' + col.replace('_',' ').title())

    plt.xticks(rotation=rotation)

    # Show the plot
    plt.show();

def replace_pattern(pattern, text):
    """
    Funtion to replace text containing specific regex pattern with space
    """
    return re.sub(pattern, ' ', text)

def clean_text(df, text_col, patterns, pattern_identifier=None):
    """
    Function to clean the text column. 
    
    INPUT
    :df: pandas dataframe
    :text_col: the name of the column containing text to be processed, datatype string
    :patterns: a list containing regex patterns used to do various things to the text column
    :pattern_identifier: a list containing the explanation of what the regex patterns do
    
    OUTPUT
    :df: pandas dataframe with the text column already cleaned
    """
    
    #initializing tqdm for pandas
    tqdm.pandas()
    
    #converting the text column into lowercase
    df[text_col] = df[text_col].str.lower()
    
    #clean the text column using regex
    for i, pattern in enumerate(patterns):
        if pattern_identifier:
            print(pattern_identifier[i])
        else:
            print(f'Regex cleaning {i+1}')
        df[text_col] = df[text_col].progress_apply(lambda text: replace_pattern(pattern, text))
    return df

def clean_input_text(input_text, patterns, pattern_identifier=None):
    """
    Function to clean an input text. 
    
    INPUT
    :input_text: input text, datatype string
    :patterns: a list containing regex patterns used to do various things to the text column
    :pattern_identifier: a list containing the explanation of what the regex patterns do
    
    OUTPUT
    :input_text: cleaned input_text
    """
    desc_tqdm = None
    
    #convert the input text into lowercase 
    input_text = input_text.lower()
    
    #create desc_tqdm for sanity checking
    if pattern_identifier:
        desc_tqdm = pattern_identifier[i]
    else:
        desc_tqdm = "Regex cleaning " + str(i+1)
    
    #clean the input text according to the regex pattern        
    for i, pattern in tqdm(enumerate(pattern), desc=desc_tqdm, 
                            total=len(pattern), leave=True, ncols=80):
        input_text = re.sub(pattern, " ", input_text)
    return input_text

def after_subplot(ax, group_name, x_label):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, None])