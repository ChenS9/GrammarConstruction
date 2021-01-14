# -*- coding: utf-8 -*-
"""
# 1. Loading Data
"""
import os
from os import chdir, getcwd
from datetime import datetime, timedelta
# this sets the directory to the directory of where our script is located
folder_loc = os.path.dirname(os.path.realpath("__file__"))
os.chdir(folder_loc)


# =============================================================================
# SET PATHS
# =============================================================================

# File with all reviews that we want to analyze:
# reviews column needs to be named "text"
reviews_path = "all.csv"

date_data_scraped = datetime(2020, 7, 18) # we use this to subtract "a month ago" aka - 30 = review date

# Are we using a manual training csv or snorkel?
# KEEP THIS to snorkel
classify_method = "SNORKEL"  # MANUAL or SNORKEL

# If using Snorkel - # Snorkel Topic Spreadsheet
topic_path = "inputs/zomato_topics.xlsx"

# set snorkel training dataset output name (we create it in line 347 )
# output name for snorkel training 
#this can be called anything.csv -- its just incase we have an error in the script
# and you want to see the training dataset that snorkel created
snorkel_training_df_output_name = "outputs/tss.csv"

# If using manual: ( SKIP IF USING SNORKEL)
# Column names need to be: 'text', 'class_simple','sent_score'
manual_training_df = "inputs/fhs.csv"


# Folder for exporting results and train/valid.txt
# make folder to save files into
ext_path = 'outputs/'

# Name the final output file
final_output = "all_reviews_hospital_classified.csv"


# =============================================================================
# IMPORTS
# =============================================================================
if not os.path.exists(ext_path):
    os.makedirs(ext_path)

# sentiment training dataset in resources folder
sentiment_training_csv = "resources/topic_identification/sentiment_training/sentiment_training_no_stopwords.csv"

from tqdm import tqdm
tqdm.pandas()
wd = getcwd()  # lets you navigate using chdir within jupyter/spyder
chdir(wd)
import pandas as pd
import re
# Natural Language Processing packages
import nltk
import nltk.data
nltk.download('stopwords')
nltk.download('punkt')
# Download the spaCy english model
# !python -m spacy download en_core_web_sm
# !pip install snorkel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LabelingFunction
from snorkel.labeling.model import LabelModel
# sys.path.insert(0, scripts_folder)
from resources.text_cleaning.functions_text_cleaning import *
from resources.topic_identification.functions_topic_identification import fast_process_io
from resources.text_cleaning.create_training_set import *
from resources.get_date import posting_time_to_num_2, days_to_date
from resources.cleaning_functions import tokenization,\
    remove_stopwords, remove_spaces, data_preprocessing


# os.chdir(scripts_folder)
# create output folder
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# print(scripts_folder + reviews_path)
df = pd.read_csv(reviews_path, encoding='utf-8')
df['text'] = df['reviews']
del df['reviews']
# df = df[:1000]
# !pip3 install --upgrade pandas
df.head()
df = df[pd.notnull(df['text'])]
# df = df.rename(columns={"text2": "text"})


"""# Data Preprocessing
Now we are done in data exploration and go to data preprocessing steps!! <br>
There are several steps to handle the text data.
- Sentence & Word tokenization
- Handling all lower case, stopwords, puntuations
- Stopwords,Punctuation
- Spelling corrections (Not cover)
- Stemming and Lemmatization --> We will not cover this.

# Tokenization
"""
df = data_preprocessing(df)


# Snorkel
if classify_method == "SNORKEL":

    """# Take 10000 rows to train labels"""
    if len(df) >= 50000:
        df_train = df.sample(n=50000)
    else:
        df_train = df.copy()

    # Read excel sheet into dictionary
    xls = pd.ExcelFile(topic_path)
    sample = pd.read_excel(xls, 'keywords')
    match = pd.read_excel(xls, 'not_match')  # test
    # Note: for topics with no required word, fill the cell with "n"
    required = pd.read_excel(xls, 'required')
    topics = sample['topic'].values.tolist()

    """remove stopwords in the keywords and non_match words"""
    for i in range(1, len(sample.columns)):
        sample.iloc[:, i] = sample.iloc[:, i].apply(str).apply(tokenization)
        sample.iloc[:, i] = sample.iloc[:, i].apply(remove_stopwords)
        sample.iloc[:, i] = sample.iloc[:, i] .apply(remove_spaces)
        sample.iloc[:, i] = sample.iloc[:, i] .apply(
            lambda x: ''.join(i + ' ' for i in x))

    for i in range(1, len(match.columns)):
        match.iloc[:, i] = match.iloc[:, i].apply(str).apply(tokenization)
        match.iloc[:, i] = match.iloc[:, i].apply(remove_stopwords)
        match.iloc[:, i] = match.iloc[:, i] .apply(
            lambda x: ''.join(i + ' ' for i in x))

    keywords = sample.drop(['topic'], axis=1).apply(
        lambda x: ",".join(x.dropna()), axis=1).values.tolist()
    # Note: there needs to be at least one word for each topic
    not_match = match.drop(['topic'], axis=1).apply(
        lambda x: ",".join(x.dropna()), axis=1).values.tolist()
    required_word = required.drop(['topic'], axis=1).apply(
        lambda x: ",".join(x.dropna()), axis=1).values.tolist()

    keywords = [[x.rstrip() for x in y.split(",")] for y in keywords]
    keywords = [[x for x in y if x != 'nan'] for y in keywords]
    keywords = [[x for x in y if len(x) > 1] for y in keywords]

    not_match = [[x.rstrip() for x in y.split(",")] for y in not_match]
    not_match = [[x for x in y if x != 'nan'] for y in not_match]
    not_match = [[x for x in y if len(x) > 1] for y in not_match]

    required_word = [[x.rstrip() for x in y.split(",")] for y in required_word]
    required_word = [[x for x in y if x != 'nan'] for y in required_word]

    key_dict = dict(zip(topics, keywords))
    print(key_dict)
    not_match_dict = dict(zip(topics, not_match))
    print(not_match_dict)
    req_dict = dict(zip(topics, required_word))
    print(req_dict)

    """Topic labeling using Snorkel"""
    """Create Dictionary for Labels"""
    ABSTAIN = -1
    NO_TOPIC = -2
    label_dict = {}
    for i in range(len(topics)):
        label_dict[topics[i]] = i
    # print(label_dict['PETS'])

    # BELOW IS OLD CODE
    # """Using Regex for pattern maching LF functions"""
    # def all_match(x, keywords,notmatch,label):
    #     #First assign labels based on the keywords matching
    #     for word in keywords:
    #         if re.search(rf'\b{word}e?s?\b',str(x), flags=re.I):   # re.I is case insensitive matching
    #     #Second filter out those rows actually do not satisfy the topic
    #             for word in notmatch:
    #                 if re.search(rf'\b{word}e?s?\b', str(x), flags=re.I):   # re.I is case insensitive matching
    #                     return ABSTAIN
    #             return label
    #     return NO_TOPIC
    # # all_match('the air quality is good',keywords=key_dict['AC'].split(','),notmatch=not_match_dict['AC'].split(','),label=label_dict['AC'])

    # """Create loop to auto generate class for each topic LF function"""
    # def all_match_lf(keywords,notmatch,label):
    #     return LabelingFunction(
    #         name=f"keyword_{keywords[0]}",
    #         f=all_match,
    #         resources=dict(keywords=keywords, notmatch=notmatch,label=label),
    #     )
    # for i in range(len(topics)):
    #     globals()['lf_all_match_%s' % topics[i]] = all_match_lf(keywords= key_dict[topics[i]],
    #                                                             notmatch = not_match_dict[topics[i]],
    #                                                             label =label_dict[topics[i]])

    """Using Regex for pattern maching LF functions"""
    def all_match(x, keywords, notmatch, required, label):
        # First assign labels based on the keywords matching

        if len(required[0]) > 1:  # If there is a required word list
            for word in required:
                if re.search(rf'\b{word}e?s?\b', str(x), flags=re.I):
                    for word in keywords:
                        # re.I is case insensitive matching
                        if re.search(rf'\b{word}e?s?\b', str(x), flags=re.I):
                            # Second filter out those rows actually do not satisfy the topic
                            for word in notmatch:
                                # re.I is case insensitive matching
                                if re.search(rf'\b{word}e?s?\b', str(x), flags=re.I):
                                    return ABSTAIN
                            return label
                return ABSTAIN

        elif len(required[0]) == 1:  # If there is no required word (filled with "n")
            for word in keywords:
                # re.I is case insensitive matching
                if re.search(rf'\b{word}e?s?\b', str(x), flags=re.I):
                    # Second filter out those rows actually do not satisfy the topic
                    for word in notmatch:
                        # re.I is case insensitive matching
                        if re.search(rf'\b{word}e?s?\b', str(x), flags=re.I):
                            return ABSTAIN
                    return label

        return NO_TOPIC
    # all_match('the air quality is good',keywords=key_dict['AC'].split(','),notmatch=not_match_dict['AC'].split(','),label=label_dict['AC'])
    # all_match('the room is spacious',keywords=key_dict['ROOM_SIZE'],notmatch=not_match_dict['ROOM_SIZE'], required = req_dict['ROOM_SIZE'],label=label_dict['ROOM_SIZE'])

    """Create loop to auto generate class for each topic LF function"""
    def all_match_lf(keywords, notmatch, required, label):
        return LabelingFunction(
            name=f"keyword_{keywords[0]}",
            f=all_match,
            resources=dict(keywords=keywords, notmatch=notmatch,
                           required=required, label=label),
        )
    for i in range(len(topics)):
        print(i)
        print(label_dict[topics[i]])
        print(key_dict[topics[i]])
        print(not_match_dict[topics[i]])
        print(req_dict[topics[i]])


        globals()['lf_all_match_%s' % topics[i]] = all_match_lf(keywords=key_dict[topics[i]],
                                                                notmatch=not_match_dict[topics[i]],
                                                                required=req_dict[topics[i]],
                                                                label=label_dict[topics[i]])

    lfs = [0] * len(topics)
    for i in range(len(topics)):
        lfs[i] = vars()['lf_all_match_%s' % topics[i]]
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)
    # LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    # def plot_label_frequency(L):
    #     plt.hist((L != ABSTAIN).sum(axis=1), density=True, bins=range(L.shape[1]))
    #     plt.xlabel("Number of labels")
    #     plt.ylabel("Fraction of dataset")
    #     plt.show()
    # plot_label_frequency(L_train)
    print('Start topic labeling')
    label_model = LabelModel(cardinality=len(topics), verbose=True)
    # original n_epochs: 500 but it was changed because of fast execution
    label_model.fit(L_train, n_epochs=100, log_freq=50, seed=123)
    df_train["class_simple"] = label_model.predict(
        L=L_train, tie_break_policy="abstain")

    df_train = df_train[df_train["class_simple"] != NO_TOPIC]
    # Save training dataset for fasttext topic labeling
    df_train.to_csv(snorkel_training_df_output_name)

    topic_df = fast_process_io(snorkel_training_df_output_name, df, 'class_simple', ext_path,
                               0.1,  # lr_in
                               70,  # epoch_in
                               2,  # ngrams_in
                               1000,  # size_each
                               0.5,  # threshold_in
                               'multi'  # single_multi
                               )

if classify_method == "MANUAL":
    """Topic labeling using fasttext"""
    topic_df = fast_process_io(manual_training_df, df, 'class_simple', ext_path,
                               0.1,  # lr_in
                               70,  # epoch_in
                               2,  # ngrams_in
                               1000,  # size_each
                               0.5,  # threshold_in
                               'multi'  # single_multi
                               )


# topic_df = topic_df[['extra_id','text','class_simple','class_simple_pred_label', 'class_simple_pred_acc','class_simple_pred_label2', 'class_simple_pred_acc2']] # 'lat','long','avg_review',
topic_df['class_simple_pred_label'] = topic_df['class_simple_pred_label'].fillna(
    '')
topic_df['class_simple_pred_label'] = topic_df['class_simple_pred_label'].apply(
    lambda x: x.replace('__label__', ''))
topic_df['class_simple_pred_label2'] = topic_df['class_simple_pred_label2'].fillna(
    '')
topic_df['class_simple_pred_label2'] = topic_df['class_simple_pred_label2'].apply(
    lambda x: x.replace('__label__', ''))
# topic_df = df.join(topic_df)
# topic_df = pd.merge(df, topic_df, how='left', on='extra_id')
# topic_df.to_csv(scripts_folder + ext_path + "outputs/topic_df.csv")

"""Sentiment labeling using fasttext"""
print('Start sentiment labeling')
# sentiment predict ('sent_score' is the name of sentiment good/bad column in training df)
sent_df = fast_process_io(sentiment_training_csv, df, 'sent_score', ext_path,
                          .1,  # lr_in
                          70,  # epoch_in
                          2,  # ngrams_in
                          2576,  # size_each
                          0.5,  # threshold_in
                          'single'  # single_multi
                          )
# sent_df.to_csv("sentiment_df.csv")
sent_df = sent_df[['extra_id', 'sent_score_pred_label', 'sent_score_pred_acc']]

sent_df['sent_score_pred_label'] = sent_df['sent_score_pred_label'].fillna('')
sent_df['sent_score_pred_label'] = sent_df['sent_score_pred_label'].apply(
    lambda x: x.replace('__label__', ''))

# clean up sentiment
sent_df['sent_score_pred_label'] = sent_df['sent_score_pred_label'].replace(
    'good', 1)
sent_df['sent_score_pred_label'] = sent_df['sent_score_pred_label'].replace(
    'bad', -1)
sent_df['sent_score_pred_label'] = sent_df['sent_score_pred_label'].replace(
    'neutral', 0)
# #drop rows with nas
# topic_sent_df = topic_sent_df.dropna()
# join dataframes
# final_df = topic_df.join(topic_sent_df)
final_df = pd.merge(topic_df, sent_df, how='inner', on='extra_id')
print(final_df.shape)
# final_df = final_df.dropna()
# df_train[“label”] will contain ABSTAIN labels as well, therefore in order to further train our secondary classifier, we’ll have to filter them out.
# output = final_df[final_df.class_simple != ABSTAIN]
output = final_df.copy()
# label_cols = ['class_simple_pred_label','class_simple_pred_label2']
# accuracy_cols = ['class_simple_pred_acc','class_simple_pred_acc2']
# output['class_simple_pred_label_combined'] = output[label_cols].values.tolist()
# output['class_simple_pred_acc_combined'] = output[accuracy_cols].values.tolist()
output['class_simple_pred_label_acc_combined'] = output['class_simple_pred_label'] + \
    output['class_simple_pred_label2']
# label_accuracy_cols = ['class_simple_pred_label','class_simple_pred_acc','class_simple_pred_label2','class_simple_pred_acc2']
# output['class_simple_pred_label_acc_combined'] = output[label_accuracy_cols].values.tolist()
# output['class_simple_pred_label_combined'] = output['class_simple_pred_label_combined'].apply(lambda x: x for x in a_list if x != "")

# def remove_empty(label):
#     new_label = list()
#     for i in label:
#         if i != '':
#             if pd.isnull(i)==False:
#                 new_label.append(i)
#     return new_label

# def remove_empty_accuracy(accuracy):
#     new_accuracy = list()
#     for i in accuracy:
#         if i != "" or i != "nan":
#             new_accuracy.append(i)
#     return new_accuracy

# output['class_simple_pred_label_acc_combined'] = output['class_simple_pred_label_acc_combined'].apply(remove_empty)
if classify_method == "SNORKEL":
    def assign_topics(x):
        if x == "-1":
            return "NOT_RELEVANT"
        elif (x.isdigit() and x != -1):
            return topics[int(x)]
        else:
            return ""

    output['topic'] = output['class_simple_pred_label_acc_combined'].apply(
        assign_topics)  # Convert numeric label to strings

else:
    output['topic'] = output['class_simple_pred_label_acc_combined']

# output['topic'] = output['class_simple_pred_label_acc_combined'].apply(lambda x: topics[int(x)] if (x.isdigit() and x != -1)  elif (x.isdigit() and x == -1) else '') #Convert numeric label to strings

# output['class_simple_pred_acc_combined'] = output['class_simple_pred_acc_combined'].apply(remove_empty_accuracy)

# output['topic'] = output['class_simple_pred_label'].apply(lambda x: topics[int(x)] if int(x) != -1 else 'ABSTAIN') #Convert numeric label to strings


output = output[output['topic'] != 'NOT_RELEVANT']
output = output[pd.notnull(output['topic'])]

# output['review_date'] = output['review_date'].apply(posting_time_to_num_2)
# output['review_date'] = output['review_date'].apply(
#     days_to_date, date_scraped=date_data_scraped)

final_output = os.path.join(ext_path, final_output)
print(f"Writing final output to {final_output}")
output.to_csv(final_output)
