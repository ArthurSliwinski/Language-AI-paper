import pandas as pd
import numpy as np
import sklearn as sk
import scipy
import spacy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from scipy.sparse import csr_matrix
import scipy.sparse as sp

# The data is loaded into a pandas dataframe providing a representation that is nice to work with
# After loading the data into a df, any rows containing NaN values are removed, to prevent problems further down the pipeline
data = pd.read_csv(r'C:\Users\sliwi\Documents\Courses\Y3\Y3Q2\Language&AI\Assignment-ResearchPaper\lai-data\lai-data\political_leaning.csv')
data.dropna()

# First a function is defined that will eventually be used ot split the data.
def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    #This function takes a dataframe containing the reddit posts and corresponding author ID's and creates a train and test split
    # It creates bins based on the amount of posts per user, which are used for stratification of the data before splitting.
    bins = [0, 15, 50, 150, 250, float('inf')]
    labels = ['small', 'medium', 'large', 'extralarge', 'supersize']
    stratificaton_data = data.groupby('auhtor_ID').count()
    stratificaton_data['count_group'] = pd.cut(stratificaton_data['post'], bins, labels=labels)

    # In order to ensure the train and test sets do not have overlapping authors the data is split based on author id's
    # This way we overlap is prevented, but the analysis of individual documents (posts) is still possible
    author_indeces = data['auhtor_ID'].unique().tolist()

    # The sklearn train_test_split is used, with seed 4 (reproduceability), and the data is stratified based on the beforementioned bins
    split = sk.model_selection.train_test_split(author_indeces, test_size=0.2, random_state=4, stratify=stratificaton_data['count_group'])
    # split = sk.model_selection.train_test_split(author_indeces, test_size=0.2, random_state=4, stratify=stratificaton_data['political_leaning'])
    training_data = data[data['auhtor_ID'].isin(split[0])]
    test_data = data[data['auhtor_ID'].isin(split[1])]

    return training_data, test_data

# since the posts contain a lot of special characters we cannot interpret, such as cirillic alphabet, we filter every word containing any other character than the latin aplhabet
def preprocess_text(text):

    ##### Uncomment the section below when running the model for the second time, to modify the vocabulary #####
    # # This defines a list of words, and then puts it in regex format to allow us to remove words from the training vocab
    # exclusion_list = ['centrist', 'libcenter']
    # # The | is used to sepertate the words in the list in regex form, and \b ensures the matching of words
    # exclusions = r'\b(?:' + '|'.join(map(re.escape, exclusion_list)) + r')\b'

    ##### Uncomment the section above when running the model for the second time, to modify the vocabulary #####

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b[^a-z]+\b', ' ', text)  # Remove words with non-Latin characters (no accents or numbers)
    text = re.sub(r'\b\d+\w*\b', ' ', text)# Remove words containing numbers (e.g., '100k', '10s')

    ##### The line below is also to be uncommented when running the modified vocabulary model #####
    # text = re.sub(exclusions, '', text)

    ##### The line above is also to be uncommented when running the modified vocabulary model #####
    return text


# Using the split function on our data we obtian two dataframes containing the training and testing data respectively
df_train, df_test= split_data(data)

# From the split dataframes the indices are obtained, so we can split the data effectively later on
train_indices = df_train.index
test_indices = df_test.index


# Here the vectorizer is defined, opring for the TF-IDF vectorizer
# lowercase and remove english frequent words, including all unigrams we get memory error
vectorizer = TfidfVectorizer(preprocessor= preprocess_text, stop_words='english', min_df=10, max_df= 10000)
# The data is vectorized as a whole and split later on to ensure that all data represented by the same features
# The output of the vectorizer fit_transform of the dataframe column, is a sparse matrix of the csr format
X_total = vectorizer.fit_transform(data['post'])

# The sparse csr matrix is split according to the previously defined indices
X_train = X_total[train_indices, :]
X_test = X_total[test_indices,:]

# The data is saved to npz files, so it is easy to load into another .py file later on
sp.save_npz('x_train.npz', X_train)
sp.save_npz('x_test.npz', X_test)


print("Data transformation and save complete")

