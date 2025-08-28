import os
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.layers import Concatenate
from keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
from tensorflow.keras import layers
import keras


# Read the original CSV file
csv_file_path_raw = os.path.join('data', 'trainingData.csv')
df_raw = pd.read_csv(csv_file_path_raw)

# Generate a unique 7-digit ID for each row
unique_ids = np.arange(1000000, 1000000 + len(df_raw))

# Add the unique ID column to the left of the DataFrame
df_raw.insert(0, 'unique_id', unique_ids)

df_raw['unique_id'] = df_raw['unique_id'].astype(str)

# Save the new DataFrame with unique IDs to 'trainingDataNew.csv'
df_raw.to_csv('data/trainingDataNew.csv', index=False)

# Print the resulting DataFrame with unique IDs
#print("Padded DataFrame with unique IDs and mentors:")


# Read the original CSV file
csv_file_path = os.path.join('data', 'trainingDataNew.csv')
df = pd.read_csv(csv_file_path)

print(df.head())
df['unique_id'] = df['unique_id'].astype(str)

mentor_database = df[['unique_id', 'mentorName', 'menteeName']]

#Replacing the headers to remove spaces and special characters in headers of the CSV file
def process_columns(df, columns):
    new_df = pd.DataFrame()
    for column in columns:
        new_column = column
        new_df[new_column] = df[column].astype(str).str.lower().replace('[^\w\s]', '', regex=True)
    return new_df

column_to_process = df.columns

df_new = process_columns(df, column_to_process)



def preprocess_data(df, column_to_preprocess):
    # Convert NaN to empty lists and ensure strings for specific columns
    for column in column_to_preprocess:
        df[column] = df[column].apply(lambda x: [] if pd.isna(x) and isinstance(x, list) else '' if pd.isna(x) else x)
    
    return df

columns_to_preprocess = df_new.columns.tolist()

# Create a new DataFrame by applying the preprocess_data function to each column
df_new = preprocess_data(df_new, columns_to_preprocess)

# Print the new DataFrame to verify the changes
print("df_new")
print(df_new.head())

def calculate_cosine_similarity_mean(row):
    
    #Mentor and Mentee Skills
    mentor_skills_expert = row['mentorSkillsExpert']
    mentee_skills_expert = row['menteeSkillsExpert']
    mentee_skills_competent = row['menteeSkillsCompetent']
    mentee_skills_basics = row['menteeSkillsBasics']

    #Mentor and Mentee Job Title
    mentee_job_title = row['menteeJobTitle']
    mentor_job_title = row['mentorJobTitle']

    #Mentor and Mentee Personality Type
    mentee_personality_type = row['menteePersonalityType']
    mentor_personality_type = row['mentorPersonalityType']

    #Mentor and Mentee Goals
    mentee_goals = row['menteeGoals']
    mentor_goals = row['mentorSkillsExpert']    

    # Use TF-IDF to transform the text data into numerical representations
    vectorizer = TfidfVectorizer()
    vectors_skills = vectorizer.fit_transform([mentor_skills_expert, mentee_skills_expert, mentee_skills_competent, mentee_skills_basics,])
    vectors_job_title = vectorizer.fit_transform([mentee_job_title, mentor_job_title, mentor_personality_type, mentor_goals])
    vectors_goals = vectorizer.fit_transform([mentee_goals, mentor_goals])
    vectors_personality_type = vectorizer.fit_transform([mentee_personality_type, mentor_personality_type])
    
    
    # Calculate cosine similarity for goals and skills
    similarity_skills_expert = cosine_similarity(vectors_skills[0:1], vectors_skills[1:2])[0][0] # Cosine similarity for expert skills
    similarity_skills_competent = cosine_similarity(vectors_skills[0:1], vectors_skills[2:3])[0][0] # Cosine similarity for competent skills
    similarity_skills_basics = cosine_similarity(vectors_skills[0:1], vectors_skills[3:4])[0][0] # Cosine similarity for basics skills
    similarity_personality_type = cosine_similarity(vectors_personality_type)[0][1] # Cosine similarity for personality type
    similarity_goals = cosine_similarity(vectors_goals)[0][1] # Cosine similarity for goals
    similarity_job_title = cosine_similarity(vectors_job_title)[0][1]
    
    # Calculate the mean of cosine similarity for goals and skills
    mean_similarity = ((similarity_skills_expert + similarity_skills_competent + similarity_skills_basics + similarity_job_title + similarity_personality_type + similarity_goals) / 6.0) * 100
    mean_similarity = round(mean_similarity, 0)
    return mean_similarity

# Calculate the mean of cosine similarity for each row and add it as a new column
df_new['mean_cosine_similarity'] = df_new.apply(calculate_cosine_similarity_mean, axis=1)
df_new['mean_cosine_similarity'] = df_new['mean_cosine_similarity'].astype(float)

#Add the cosine similarity to the mentor_database dataframe
mentor_database['mean_cosine_similarity'] = df_new['mean_cosine_similarity']

# Calculate the mean of the 'values_float' column in the new DataFrame
mean_cosine_value = df_new['mean_cosine_similarity'].mean()

# Convert entire DataFrame to string
df_new = df_new.astype(str)

print(df_new.head())

print('Cosine Mean: ', mean_cosine_value)

def tokenize_text_columns(df, columns):
    new_df = pd.DataFrame()
    tokenizer = Tokenizer()
    
    for column in columns:
        if column == 'unique_id': #Skip tokenising the 'unique_id' column
            new_df[column] = df[column] #.apply(lambda x: [x])
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

print(df_new.dtypes)

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

#Dividing the data into training and testing data

# Calculate the number of rows for training and testing data
total_rows = padded_df.shape[0]

# Calculate the number of rows for training data (80% of total rows)
train_rows = int(total_rows * 0.8)

# Calculate the number of rows for testing data (20% of total rows)
test_rows = total_rows - train_rows

# Split the DataFrame into training and testing data
padded_df_train = padded_df.iloc[:train_rows]
padded_df_test = padded_df.iloc[train_rows:]

num_rows_train = len(padded_df_train)
print('Number of rows for training:', num_rows_train)

num_rows_test = len(padded_df_test)
print('Number of rows for test:', num_rows_test)

#Machine Learning Model

# Define the number of preference features

# Function to preprocess the preference data
def preprocess_preferences(preferences):
    return np.array(preferences)

# Load and preprocess the dataset
def load_dataset(df):
    
    selected_columns = [k for k in df.columns if (k[:16]=='tokenized_mentee' or k[:16]=='tokenized_mentor'
                                                 or k[:15]=='tokenized_mean_')]
    df_new = df[selected_columns]
    preferences = df_new.values
    
    # DEFINE THE INPUTE LAYER
    selected_columns_skills_goals = [k for k in df.columns if (k[:20]=='tokenized_menteeGoal'
                                                               or k[:22]=='tokenized_menteeSkills'
                                                              or k[:22]=='tokenized_mentorSkills')]
    df_new_skills_goals = df[selected_columns_skills_goals]
    skills_goals = df_new_skills_goals.values

    #DEFINE THE PERSONALITY LAYER
    selected_columns_personality = [k for k in df.columns if (k[:27]=='tokenized_menteePersonality'
                                                               or k[:27]=='tokenized_mentorPersonality')]
    df_new_personality = df[selected_columns_personality]
    personality = df_new_personality.values
    
    return preferences, skills_goals, personality

def load_labels_cosine(df):
    
    # DEFINE THE LABELS/OUTPUT    
    selected_columns_labels = [k for k in df.columns if (k[:15]=='tokenized_mean_')]
    df_new_labels = df[selected_columns_labels]
    labels = df_new_labels.values
    

    return labels

#selected_columns = [k for k in padded_df_train.columns if k.startswith('token')]
#df_new_selected_column = padded_df_train[selected_columns]

#num_rows_selected_column = len(df_new_selected_column.columns)
#print('Number of rows - selected column:', num_rows_selected_column)

#Columns for skills and goals
#selected_columns_skills_goals = [k for k in padded_df_train.columns if any(word in k.lower() for word in ['skill', 'goal'])]
#df_new_skills_goals = padded_df_train[selected_columns_skills_goals]
#num_rows_skills_goals = len(df_new_skills_goals.columns)
#print('Number of rows - skills and goals:', num_rows_skills_goals)

#Columns for personality
#selected_columns_personality = [k for k in padded_df_train.columns if any(word in k.lower() for word in ['personality', 'Personality'])]
#df_new_personality = padded_df_train[selected_columns_personality]
#num_rows_personality = len(df_new_personality.columns)
#print('Number of rows - personality:', num_rows_personality)

#Columns for labels
#selected_columns_labels = [k for k in padded_df_train.columns if 'mean_' in k]
#df_new_labels = padded_df_train[selected_columns_labels]

#num_rows_labels = len(df_new_labels.columns)
#print('Number of rows - labels:', num_rows_labels)

#Tuning the Epoch and create CNN model 
def create_cnn_model_with_two_labels(num_skills_goals_features, num_personality_features, num_preference_features):
    preferences_input = layers.Input(shape=(num_preference_features,))
    skills_goals_input = layers.Input(shape=(num_skills_goals_features,))
    personality_input = layers.Input(shape=(num_personality_features,)) 

    # Dense layers for preference processing
    preference_layers = layers.Dense(64, activation='relu')(preferences_input)
    preference_layers = layers.Dense(32, activation='relu')(preference_layers)
    
    # Skills/Goals processing layers
    skills_goals_layers = layers.Dense(16, activation='relu')(skills_goals_input)
    personality_layers = layers.Dense(16, activation='relu')(personality_input)
    
    # Concatenate preference output and skills/goals input
    combined_layers = layers.concatenate([preference_layers, skills_goals_layers, personality_layers])

    # Final dense layers for classification
    combined_layers = layers.Dense(32, activation='relu')(combined_layers)

    # Output layers for each label
    output_cosine = layers.Dense(26, activation='sigmoid', name='cosine_similarity')(combined_layers)

    model = keras.Model(inputs=[preferences_input, skills_goals_input, personality_input],
                        outputs=[output_cosine])
    
    model.compile(optimizer='adam',
                  loss={'cosine_similarity': 'binary_crossentropy'},
                  metrics={'cosine_similarity': 'accuracy'})
    
    return model

# Load and split the dataset 
preferences, skills_goals, personality = load_dataset(padded_df_train)

labels_cosine = load_labels_cosine(padded_df_train)

print('preferences:', preferences.shape[1])
print('skills_goals:', skills_goals.shape[1])
print('personality:', personality.shape[1])

# Train the model with both labels


# Train the model with both labels
def train_model(skills_goals, personality, preferences, labels_cosine):
    # Create the CNN model with two labels
    num_skills_goals_features = skills_goals.shape[1]
    num_personality_features = personality.shape[1]
    num_preference_features = preferences.shape[1]

    model = create_cnn_model_with_two_labels(num_skills_goals_features, num_personality_features, num_preference_features)

    # Train the model with both labels
    history = model.fit([preferences, skills_goals, personality], 
                        {"cosine_similarity": labels_cosine},
                        epochs=15, batch_size=32, validation_split=0.2)

    # Extract training and validation MAE from the history object
    train_cosine_mae = history.history['accuracy']
    val_cosine_mae = history.history['val_accuracy']

    return history, train_cosine_mae, val_cosine_mae, model


# Call the train_model function and assign the returned values to variables
history, train_cosine_mae, val_cosine_mae, model = train_model(skills_goals, personality, preferences, labels_cosine)

# Now you can use the train_cosine_mae variable
print("Training Cosine Similarity Accuracy:", train_cosine_mae)
mean_train_cosine_accuracy = np.mean(train_cosine_mae)
mean_val_cosine_accuracy = np.mean(val_cosine_mae)

print("Mean Training Cosine Accuracy:", mean_train_cosine_accuracy)
print("Mean Validation Cosine Accuracy:", mean_val_cosine_accuracy)

# Create an array representing the number of epochs
epochs = range(1, len(train_cosine_mae) + 1)

# Save the model
model.save("mentor_recommendation_model_with_personalities_v2.1.h5")
print("Model saved as 'mentor_recommendation_model_with_personalities_v2.1.h5'.")

test_preferences, test_skills_goals, test_personality = load_dataset(padded_df_test)

test_labels_cosine = load_labels_cosine(padded_df_test)

print('test_preferences:', test_preferences.shape[1])
print('test_skills_goals:', test_skills_goals.shape[1])
print('test_personality:', test_personality.shape[1])


# Load the saved model
model = keras.models.load_model("mentor_recommendation_model_with_personalities_v2.1.h5")


# Make predictions
predictions = model.predict([test_preferences, test_skills_goals, test_personality])

# Threshold for binary predictions
threshold = 0.7

# Convert predictions to binary values (0 or 1) for each label
predicted_classes_cosine = [(prediction > threshold).astype(int) for prediction in predictions]

accuracy_cosine = np.mean([np.mean(predicted_class == test_labels_cosine) for predicted_class, test_labels_cosine in zip(predicted_classes_cosine, test_labels_cosine)])
print("Average Model Accuracy (Cosine Similarity):", accuracy_cosine)

accuracy_values = ([np.mean(predicted_class == test_labels_cosine) for predicted_class, test_labels_cosine in zip(predicted_classes_cosine, test_labels_cosine)])

# Convert predictions to float values for each label
# Function to convert each array in the list to a single float value using the mean
def convert_arrays_to_floats(arrays):
    float_values = [np.mean(array) for array in arrays]
    return float_values

# Convert using mean
predicted_classes_cosine = convert_arrays_to_floats(predicted_classes_cosine)
print("Float values (mean):", predicted_classes_cosine)
actual_classes_cosine = convert_arrays_to_floats(test_labels_cosine)

padded_df_test['predicted_cosine_similarity'] = predicted_classes_cosine
padded_df_test['actual'] = actual_classes_cosine
padded_df_test['accuracy_values'] = accuracy_values
#print(padded_df_test.head())
# Print the average accuracy for each label

merged_df = padded_df_test.merge(mentor_database[['unique_id','mentorName', 'mean_cosine_similarity']], on='unique_id', how='left')


final_df = merged_df[['unique_id','mentorName','mean_cosine_similarity', 'predicted_cosine_similarity', 'actual', 'accuracy_values']]

print(final_df)