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
csv_file_path = os.path.join('data', 'trainingData.csv')
df = pd.read_csv(csv_file_path)

# Output column names and some data for verification
print("Column names:")
print(df.columns)
print("\nSample data:")
print(df.head())

#Replacing the headers to remove spaces and special characters in headers of the CSV file
def process_columns(df, columns):
    new_df = pd.DataFrame()
    for column in columns:
        new_column = column
        new_df[new_column] = df[column].str.lower().replace('[^\w\s]', '', regex=True)
    return new_df

column_to_process = df.columns

df_new = process_columns(df, column_to_process)

print(df_new.head())


# Convert entire DataFrame to string
df_new2 = df_new.astype(str)
print(df_new2.dtypes)

# Convert NaN to empty lists and ensure strings
df_new2['menteeSkillsExpert'] = df_new2['menteeSkillsExpert'].apply(lambda x: [] if pd.isna(x) else x)
df_new2['menteeSkillsCompetent'] = df_new2['menteeSkillsCompetent'].apply(lambda x: [] if pd.isna(x) else x)
df_new2['menteeGoals'] = df_new2['menteeGoals'].apply(lambda x: [] if pd.isna(x) else x)
df_new2['menteePersonalityType'] = df_new2['menteePersonalityType'].apply(lambda x: '' if pd.isna(x) else x)
df_new2['mentorSkillsExpert'] = df_new2['mentorSkillsExpert'].apply(lambda x: [] if pd.isna(x) else x)
df_new2['mentorPersonalityType'] = df_new2['mentorPersonalityType'].apply(lambda x: '' if pd.isna(x) else x)
df_new2['menteeJobTitle'] = df_new2['mentorJobTitle'].apply(lambda x: '' if pd.isna(x) else x)
#df_new2['mentorPersonalityType'] = df_new2['mentorPersonalityType'].apply(lambda x: '' if pd.isna(x) else x)
#df_new2['mentorPersonalityType'] = df_new2['mentorPersonalityType'].apply(lambda x: '' if pd.isna(x) else x)

print(df_new2.head())

# Adding y2 - Cosine Similarities

def calculate_cosine_similarity_mean(row):
    
    #Mentor and Mentee Skills
    mentor_skills_expert = row['mentorSkillsExpert']
    mentee_skills_expert = row['menteeSkillsExpert']
    mentee_skills_competent = row['menteeSkillsCompetent']
    mentee_skills_basics = row['menteeSkillsBasics']

    #Mentor and Mentee Job Title
    mentee_job_title = row['menteeJobTitle']
    mentor_job_title = row['mentorJobTitle']
    

    # Use TF-IDF to transform the text data into numerical representations
    vectorizer = TfidfVectorizer()
    vectors_skills = vectorizer.fit_transform([mentor_skills_expert, mentee_skills_expert, mentee_skills_competent, mentee_skills_basics])
    vectors_job_title = vectorizer.fit_transform([mentee_job_title, mentor_job_title])
    
    
    # Calculate cosine similarity for goals and skills
    similarity_skills_expert = cosine_similarity(vectors_skills[0:1], vectors_skills[1:2])[0][0] # Cosine similarity for goals
    similarity_skills_competent = cosine_similarity(vectors_skills[0:1], vectors_skills[2:3])[0][0] # Cosine similarity for goals
    similarity_skills_basics = cosine_similarity(vectors_skills[0:1], vectors_skills[3:4])[0][0] # Cosine similarity for goals
    
    similarity_job_title = cosine_similarity(vectors_job_title)[0][1]
    
    # Calculate the mean of cosine similarity for goals and skills
    mean_similarity = ((similarity_skills_expert + similarity_skills_competent + similarity_skills_basics + similarity_job_title) / 4.0) * 100
    mean_similarity = round(mean_similarity, 2)
    return mean_similarity

# Calculate the mean of cosine similarity for each row and add it as a new column
df_new2['mean_cosine_similarity'] = df_new2.apply(calculate_cosine_similarity_mean, axis=1)

print(df_new2.head())

# Create a new DataFrame and copy the existing column as float
mean_df = pd.DataFrame()
mean_df['mean_cosine_similarity'] = df_new2['mean_cosine_similarity'].astype(float)  # Convert to float and add to new DataFrame

# Calculate the mean of the 'values_float' column in the new DataFrame
mean_cosine_value = mean_df['mean_cosine_similarity'].mean()

print('Cosine Mean: ', mean_cosine_value)

# Convert entire DataFrame to string
df_new2 = df_new2.astype(str)
print(df_new2.dtypes)

#Tokenise the string columns to sequences

def tokenize_text_columns(df, columns):
    new_df = pd.DataFrame()
    tokenizer = Tokenizer()
    
    for column in columns:
        new_column = 'tokenized_' + column
        texts = df[column].tolist()
        
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, padding='post')
        
        new_column = 'tokenized_' + column
        new_df[new_column] = padded_sequences.tolist()
    
    return new_df

columns_to_tokenize = df_new2.columns

df_tok = tokenize_text_columns(df_new2, columns_to_tokenize)

print(df_tok.head())

print(df_tok.dtypes)

# Find the maximum list length in all columns
max_length = max(df_tok.applymap(len).max())

# Initialize an empty DataFrame to store the padded columns
padded_df = pd.DataFrame()

# Loop through each column in the original DataFrame
for col_name in df_tok.columns:
    # Get the column data
    column_data = df_tok[col_name]
    
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

#Number of training and testing data
padded_df_1 = padded_df.iloc[:85]
padded_df_test = padded_df.iloc[85:]

num_rows_train = len(padded_df_1)
print('Number of rows for training:', num_rows_train)

num_rows_test = len(padded_df_test)
print('Number of rows for test:', num_rows_test)

#Machine Learning Model 

# Set the seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the number of preference features
NUM_PREFERENCE_FEATURES = 312 #8

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

    # DEFINE THE PERSONALITY LAYER
    #selected_columns_personality = [k for k in df.columns if (k[:27]=='tokenized_menteePersonality'
                                                               #or k[:27]=='tokenized_mentorPersonality')]
    #df_new_personality = df[selected_columns_personality]
    #personality = df_new_personality.values
    
    return preferences, skills_goals #, personality

def load_labels_jaccard(df):
    
    # DEFINE THE LABELS/OUTPUT    
    selected_columns_labels = [k for k in df.columns if (k[:15]=='tokenized_match')]
    df_new_labels = df[selected_columns_labels]
    labels = df_new_labels.values
    

    return labels

def load_labels_cosine(df):
    
    # DEFINE THE LABELS/OUTPUT    
    selected_columns_labels = [k for k in df.columns if (k[:15]=='tokenized_mean_')]
    df_new_labels = df[selected_columns_labels]
    labels = df_new_labels.values
    

    return labels

selected_columns = [k for k in padded_df_1.columns if (k[:16]=='tokenized_mentee' 
                                                       or k[:16]=='tokenized_mentor'
                                                       or k[:15]=='tokenized_match' 
                                                       or k[:15]=='tokenized_mean_')]
df_new_selected_column = padded_df_1[selected_columns]

num_rows_selected_column = len(df_new_selected_column.columns)
print('Number of rows - selected column:', num_rows_selected_column)

#Columns for skills and goals

selected_columns_skills_goals = [k for k in padded_df_1.columns if (k[:20]=='tokenized_menteeGoal'
                                                               or k[:22]=='tokenized_menteeSkills'
                                                                   or k[:22]=='tokenized_mentorSkills')]
df_new_skills_goals = padded_df_1[selected_columns_skills_goals]

num_rows_skills_goals = len(df_new_skills_goals.columns)
print('Number of rows - skills and goals:', num_rows_skills_goals)

#Columns for labels
selected_columns_labels = [k for k in padded_df_1.columns if (k[:15]=='tokenized_mean_')]
df_new_labels = padded_df_1[selected_columns_labels]

num_rows_labels = len(df_new_labels.columns)
print('Number of rows - labels:', num_rows_labels)

#Tuning the Epoch and create CNN model 
def create_cnn_model_with_two_labels():
    preferences_input = layers.Input(shape=(NUM_PREFERENCE_FEATURES,))
    skills_goals_input = layers.Input(shape=(130,)) #7

    # Dense layers for preference processing
    preference_layers = layers.Dense(64, activation='relu')(preferences_input)
    preference_layers = layers.Dense(32, activation='relu')(preference_layers)
    
    # Skills/Goals processing layers
    skills_goals_layers = layers.Dense(16, activation='relu')(skills_goals_input)
    
    # Concatenate preference output and skills/goals input
    combined_layers = layers.concatenate([preference_layers, skills_goals_layers])

    # Final dense layers for classification
    combined_layers = layers.Dense(32, activation='relu')(combined_layers)

    # Output layers for each label
    output_cosine = layers.Dense(26, activation='sigmoid', name='cosine_similarity')(combined_layers)

    model = keras.Model(inputs=[preferences_input, skills_goals_input],
                        outputs=[output_cosine])
    
    model.compile(optimizer='adam',
                  loss={'cosine_similarity': 'binary_crossentropy'},
                  metrics={'cosine_similarity': 'accuracy'})
    
    return model
# Load and split the dataset 
preferences, skills_goals = load_dataset(padded_df_1)

labels_jaccard = load_labels_jaccard(padded_df_1)

labels_cosine = load_labels_cosine(padded_df_1)

print('preferences:', preferences.shape[1])
print('skills_goals:', skills_goals.shape[1])


# Train the model with both labels
model = create_cnn_model_with_two_labels()
history = model.fit([preferences, skills_goals], 
                    {"cosine_similarity": labels_cosine},
                    epochs=15, batch_size=32, validation_split=0.2)

print(history.history.keys())

# Extract training and validation MAE from the history object

train_cosine_mae = history.history['accuracy']
val_cosine_mae = history.history['val_accuracy']

# Create an array representing the number of epochs
epochs = range(1, len(train_cosine_mae) + 1)

# Plotting training and validation MAE for Cosine Similarity
plt.subplot(1, 2, 2)
plt.plot(epochs, train_cosine_mae, 'bo', label='Training Cosine Accuracy')
plt.plot(epochs, val_cosine_mae, 'r', label='Validation Cosine Accuracy')
plt.title('Training and Validation Cosine Accuracy Without Personality')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save("mentor_recommendation_model_without_personalities.h5")
print("Model saved as 'mentor_recommendation_model_without_personalities.h5'.")

# Calculate mean accuracy for cosine_similarity
mean_train_cosine_accuracy = np.mean(train_cosine_mae)
mean_val_cosine_accuracy = np.mean(val_cosine_mae)

print("Mean Training Cosine Accuracy:", mean_train_cosine_accuracy)
print("Mean Validation Cosine Accuracy:", mean_val_cosine_accuracy)

test_preferences, test_skills_goals = load_dataset(padded_df_test)

test_labels_jaccard = load_labels_jaccard(padded_df_test)

test_labels_cosine = load_labels_cosine(padded_df_test)

print('test_preferences:', test_preferences.shape[1])
print('test_skills_goals:', test_skills_goals.shape[1])


# Load the saved model
model = keras.models.load_model("mentor_recommendation_model_without_personalities.h5")

# Make predictions
predictions = model.predict([test_preferences, test_skills_goals])

# Threshold for binary predictions
threshold = 0.5

# Convert predictions to binary values (0 or 1) for each label
predicted_classes_cosine = [(prediction > threshold).astype(int) for prediction in predictions[1]]

# Calculate accuracy for each label separately
accuracy_cosine = np.mean([np.mean(predicted_class == test_labels_cosine) for predicted_class, test_labels_cosine in zip(predicted_classes_cosine, test_labels_cosine)])

# Print the average accuracy for each label
print("Average Model Accuracy (Cosine Similarity):", accuracy_cosine)

