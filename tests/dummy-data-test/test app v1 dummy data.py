import os
import pandas as pd  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers import Concatenate
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras import layers
import keras

# Insert the Data and create a dataframe

mentee_database = pd.read_csv(
    'tests/dummy-data-test/mentee-dummy-data.csv')
mentor_database = pd.read_csv(
    'tests/dummy-data-test/mentor-dummy-data.csv')

# Adding a prefix mentor_ to all the column headers of mentor database
mentor_database.columns = ['mentor_' + col for col in mentor_database.columns]

print("Mentee Database:")
print(mentee_database.head())
print("Mentor Database:")
print(mentor_database.head())

# Merge the dataframes
df = mentee_database.merge(mentor_database, how='cross')

# Display the final DataFrame
# print(df)

# Generate a unique 7-digit ID for each row
unique_ids = np.arange(1000000, 1000000 + len(df))

# Add the unique ID column to the left of the DataFrame
df.insert(0, 'unique_id', unique_ids)

df['unique_id'] = df['unique_id'].astype(str)

# Save the new DataFrame with unique IDs to 'trainingDataNew.csv'
df.to_csv('tests/dummy-data-test/dummyDataMerged.csv', index=False)
# print(df.head())

# Saving the main database
mentor_database = df[['unique_id', 'fullName',
                      'emailAddress', 'mentor_fullName', 'mentor_emailAddress']]
print(mentor_database.head())

# Preprocess the data

# Replacing the headers to remove spaces and special characters in headers of the CSV file


def process_columns(df: pd.DataFrame, columns: pd.DataFrame):
    new_df = pd.DataFrame()
    for column in columns:
        new_column = column
        new_df[new_column] = df[column].astype(
            str).str.lower().replace('[^\w\s]', '', regex=True)
    return new_df


column_to_process = df.columns

df_new = process_columns(df, column_to_process)


def preprocess_data(df: pd.DataFrame, column_to_preprocess: pd.DataFrame):
    # Convert NaN to empty lists and ensure strings for specific columns
    for column in column_to_preprocess:
        df[column] = df[column].apply(lambda x: [] if pd.isna(
            x) and isinstance(x, list) else '' if pd.isna(x) else x)

    return df


columns_to_preprocess = df_new.columns.tolist()

# Create a new DataFrame by applying the preprocess_data function to each column
df_new = preprocess_data(df_new, columns_to_preprocess)

# Print the new DataFrame to verify the changes
print("df_new")
print(df_new.head())

# Tokenize the data


def tokenize_text_columns(df: pd.DataFrame, columns: pd.DataFrame):
    new_df = pd.DataFrame()
    tokenizer = Tokenizer()

    for column in columns:
        if column == 'unique_id':  # Skip tokenising the 'unique_id' column
            new_df[column] = df[column]  # .apply(lambda x: [x])
        else:
            new_column = 'tokenized_' + column
            texts = df[column].tolist()

            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            padded_sequences = pad_sequences(sequences, padding='post')

            new_column = 'tokenized_' + column
            new_df[new_column] = padded_sequences.tolist()

    return new_df


columns_to_tokenize = df_new.columns

df_new = tokenize_text_columns(df_new, columns_to_tokenize)

print(df_new.head())

# Find the maximum list length in all columns
max_length = max(df_new.applymap(len).max())

# Initialize an empty DataFrame to store the padded columns
padded_df = pd.DataFrame()

for col_name in df_new.columns:
    if col_name == 'unique_id':
        padded_df[col_name] = df_new[col_name]

    else:
        # Get the column data
        column_data = df_new[col_name]

        # Initialize an empty list to store the padded values
        padded_values = []

        # Iterate through each element in the column
        for value in column_data:
            # Pad the list with zeros to reach the maximum length
            padded_value = value + [0] * (max_length - len(value))
            padded_values.append(padded_value)

        # Create new column names based on the original column name
        new_col_names = [f'{col_name}_{idx}' for idx in range(max_length)]

        # Create a DataFrame from the padded values and new column names
        padded_col_df = pd.DataFrame(padded_values, columns=new_col_names)

        # Concatenate the padded DataFrame with the main padded DataFrame
        padded_df = pd.concat([padded_df, padded_col_df], axis=1)

# Print the resulting padded DataFrame
print(padded_df.head)


# Load the model

NUM_PREFERENCE_FEATURES = padded_df.shape[1] - 1


def load_dataset(df: pd.DataFrame):

    selected_columns = [k for k in df.columns if (k[:9] == 'tokenized')]
    df_new = df[selected_columns]
    preferences = df_new.values

    # DEFINE THE INPUTE LAYER
    selected_columns_skills_goals = [k for k in df.columns if (k[:22] == 'tokenized_longTermGoal'
                                                               or k[:29] == 'tokenized_mentor_longTermGoal'
                                                               or k[:33] == 'tokenized_firstBasicIndustrySkill'
                                                               or k[:29] == 'tokenized_firstBasicSoftSkill'
                                                               or k[:34] == 'tokenized_firstExpertIndustrySkill'
                                                               or k[:40] == 'tokenized_mentor_firstBasicIndustrySkill'
                                                               or k[:36] == 'tokenized_mentor_firstBasicSoftSkill'
                                                               or k[:41] == 'tokenized_mentor_firstExpertIndustrySkill')]
    df_new_skills_goals = df[selected_columns_skills_goals]
    skills_goals = df_new_skills_goals.values

    # DEFINE THE PERSONALITY LAYER
    selected_columns_personality = [k for k in df.columns if (k[:25] == 'tokenized_personalityType'
                                                              or k[:32] == 'tokenized_mentor_personalityType')]
    df_new_personality = df[selected_columns_personality]
    personality = df_new_personality.values

    return preferences, skills_goals, personality

# Load the trained model


def pad_or_truncate(arr: pd.DataFrame, target_shape: pd.DataFrame):
    if arr.shape[1] > target_shape:
        return arr[:, :target_shape]
    else:
        return np.pad(arr, ((0, 0), (0, target_shape - arr.shape[1])), mode='constant')


# Load the saved model
model: Model = keras.models.load_model(
    "src/mentor_recommendation_model_with_personalities_v2.1.h5")

# Preprocess the new dataset (tokenize, pad, etc.)
# Assuming new_preferences, new_skills_goals, and new_personality are the new dataset
new_preferences, new_skills_goals, new_personality = load_dataset(padded_df)

# Adjust the target shape as per model input
new_preferences = pad_or_truncate(new_preferences, 312)
new_skills_goals = pad_or_truncate(new_skills_goals, 130)
new_personality = pad_or_truncate(new_personality, 52)

threshold = 0.5
# Make predictions
predictions = model.predict(
    [new_preferences, new_skills_goals, new_personality])

# Convert predictions to binary values (0 or 1) for each label
predicted_classes_cosine = [(prediction > threshold).astype(int)
                            for prediction in predictions]

# Print the predicted classes
print("Predicted Classes (Cosine Similarity):", predicted_classes_cosine)


def convert_arrays_to_floats(arrays):
    float_values = [np.mean(array) for array in arrays]
    return float_values


# Convert using mean
predicted_classes_cosine = convert_arrays_to_floats(predicted_classes_cosine)
print("Float values (mean):", predicted_classes_cosine)

padded_df['predicted_cosine_similarity'] = predicted_classes_cosine

merged_df = padded_df.merge(mentor_database[[
                            'unique_id', 'mentor_fullName', 'fullName']], on='unique_id', how='left')


final_df = merged_df[['unique_id', 'fullName',
                      'mentor_fullName', 'predicted_cosine_similarity']]
# Sort the final_df DataFrame by 'fullName' in ascending order and then by 'predicted_cosine_similarity' in ascending order
# final_df = final_df.sort_values(by=['fullName', 'predicted_cosine_similarity'], ascending=[True, True])

# Print the sorted DataFrame
print("The Initial Output")
# print(final_df.head())
print(final_df)

# final_df.to_csv('proposed_matches.csv', index=False)

# Sort the final_df DataFrame by 'fullName' and 'predicted_cosine_similarity' in descending order
final_df = final_df.sort_values(
    by=['fullName', 'predicted_cosine_similarity'], ascending=[True, False])

# Group the DataFrame by 'fullName'
grouped_df = final_df.groupby('fullName')

# Select the top 3 rows for each group
top_3_df = grouped_df.head(3)

# Print the top 3 DataFrame
print("Top 3 Mentors per Mentee:")
print(top_3_df)
top_3_df.to_csv('tests/dummy-data-test/proposed_matches.csv', index=False)
