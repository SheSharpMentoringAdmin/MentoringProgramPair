import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd

# Define a function to create the model
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

    model = Model(inputs=[preferences_input, skills_goals_input, personality_input], outputs=[output_cosine])
    
    model.compile(optimizer='adam', loss={'cosine_similarity': 'binary_crossentropy'}, metrics={'cosine_similarity': 'accuracy'})
    
    return model

# Function to load and preprocess dataset
def load_and_preprocess_dataset(df):
    preferences = np.array(df['preferences'].tolist())
    skills_goals = np.array(df['skills_goals'].tolist())
    personality = np.array(df['personality'].tolist())
    labels_cosine = np.array(df['labels_cosine'].tolist())
    return preferences, skills_goals, personality, labels_cosine

# Function to pad/truncate the input data to a specific shape
def pad_or_truncate(arr, target_shape):
    if arr.shape[1] > target_shape:
        return arr[:, :target_shape]
    else:
        return np.pad(arr, ((0, 0), (0, target_shape - arr.shape[1])), mode='constant')

# Function to train the model
def train_model(skills_goals, personality, preferences, labels_cosine):
    num_skills_goals_features = skills_goals.shape[1]
    num_personality_features = personality.shape[1]
    num_preference_features = preferences.shape[1]

    model = create_cnn_model_with_two_labels(num_skills_goals_features, num_personality_features, num_preference_features)

    history = model.fit([preferences, skills_goals, personality], 
                        {"cosine_similarity": labels_cosine},
                        epochs=15, batch_size=32, validation_split=0.2)

    train_cosine_acc = history.history['cosine_similarity_accuracy']
    val_cosine_acc = history.history['val_cosine_similarity_accuracy']

    return history, train_cosine_acc, val_cosine_acc, model

# Example usage with dummy data
padded_df_train = pd.DataFrame({
    'preferences': [np.random.rand(112).tolist() for _ in range(100)],
    'skills_goals': [np.random.rand(7).tolist() for _ in range(100)],
    'personality': [np.random.rand(70).tolist() for _ in range(100)],
    'labels_cosine': [np.random.randint(2, size=26).tolist() for _ in range(100)]
})

preferences, skills_goals, personality, labels_cosine = load_and_preprocess_dataset(padded_df_train)

# Train the model with the preprocessed data
history, train_cosine_acc, val_cosine_acc, model = train_model(skills_goals, personality, preferences, labels_cosine)

print("Training Cosine Similarity Accuracy:", train_cosine_acc)
mean_train_cosine_accuracy = np.mean(train_cosine_acc)
mean_val_cosine_accuracy = np.mean(val_cosine_acc)

print("Mean Training Cosine Accuracy:", mean_train_cosine_accuracy)
print("Mean Validation Cosine Accuracy:", mean_val_cosine_accuracy)

# Save the model
model.save("mentor_recommendation_model_with_personalities_v2.h5")
print("Model saved as 'mentor_recommendation_model_with_personalities_v2.h5'.")

# Prediction with a different shape dataset
padded_df_test = pd.DataFrame({
    'preferences': [np.random.rand(14).tolist() for _ in range(20)],
    'skills_goals': [np.random.rand(462).tolist() for _ in range(20)],
    'personality': [np.random.rand(464).tolist() for _ in range(20)]
})

# Preprocess the test data
preferences_test, skills_goals_test, personality_test, _ = load_and_preprocess_dataset(padded_df_test)

# Pad or truncate the test data to match the model input shape
preferences_test = pad_or_truncate(preferences_test, 10)  # Adjust the target shape as per model input
skills_goals_test = pad_or_truncate(skills_goals_test, 7)
personality_test = pad_or_truncate(personality_test, 7)

# Predict with the model
predictions = model.predict([preferences_test, skills_goals_test, personality_test])
print("Predictions:", predictions)
