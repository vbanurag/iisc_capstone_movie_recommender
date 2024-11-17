from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import numpy as np

class MovieRecommenderModel:
    def __init__(self, n_users: int, n_movies: int, refind_df, n_factors: int = 150):
        self.n_users = n_users
        self.n_movies = n_movies
        self.refined_dataset = refind_df
        self.n_factors = n_factors
        self.model = self.build_model()

    def build_model(self) -> keras.Model:
        """Build the neural network architecture"""
        # User input and embedding
        user = keras.layers.Input(shape=(1,), dtype='int32', name='user_input')
        u = keras.layers.Embedding(
            input_dim=self.n_users,
            output_dim=self.n_factors,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )(user)
        u = keras.layers.Reshape((self.n_factors,))(u)

        # Movie input and embedding
        movie = keras.layers.Input(shape=(1,), dtype='int32')
        m = keras.layers.Embedding(
            input_dim=self.n_movies,
            output_dim=self.n_factors,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )(movie)
        m = keras.layers.Reshape((self.n_factors,))(m)

        # Merge layers and add dense layers
        x = keras.layers.Concatenate()([u, m])
        x = keras.layers.Dropout(0.05)(x)

        # First dense layer
        x = keras.layers.Dense(32, kernel_initializer='he_normal')(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Dropout(0.05)(x)

        # Second dense layer
        x = keras.layers.Dense(16, kernel_initializer='he_normal')(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Dropout(0.05)(x)

        # Output layer
        x = keras.layers.Dense(9)(x)
        output = keras.layers.Activation(activation='softmax')(x)

        model = keras.Model(inputs=[user, movie], outputs=output)
        
        model.compile(
            optimizer=keras.optimizers.Adagrad(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        return model

    def train(self, X_train: Tuple, y_train: np.ndarray, 
              X_val: Tuple, y_val: np.ndarray, 
              epochs: int = 80, batch_size: int = 128) -> keras.callbacks.History:
        """Train the model with callbacks for early stopping and checkpointing"""
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.75,
            patience=3,
            min_lr=0.00001,
            verbose=1,
            min_delta=1e-4
        )

        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath='movie_recommend_dd_softmax.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        # Train the model
        history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, y_val),
            shuffle=True,
            callbacks=[reduce_lr, checkpoint]
        )

        return history
    
    def recommender_movie_system(self, user_id, n_movies = 10):
        print("")
        print("Movie seen by the User:")
        pprint(list(self.refined_dataset[self.refined_dataset['user id'] == user_id]['movie title']))
        print("")
        model = self.model
        encoded_user_id = self.user_enc.transform([user_id])

        seen_movies = list(self.refined_dataset[self.refined_dataset['user id'] == user_id]['movie'])
        unseen_movies = [i for i in range(min(self.refined_dataset['movie']), max(self.refined_dataset['movie'])+1) if i not in seen_movies]
        model_input = [np.asarray(list(encoded_user_id)*len(unseen_movies)), np.asarray(unseen_movies)]
        predicted_ratings = model.predict(model_input)
        predicted_ratings = np.max(predicted_ratings, axis=1)
        sorted_index = np.argsort(predicted_ratings)[::-1]
        recommended_movies = self.movie_enc.inverse_transform(sorted_index)
        print("---------------------------------------------------------------------------------")
        print("Top "+str(n_movies)+" Movie recommendations for the User "+str(user_id)+ " are:")
        pprint(list(recommended_movies[:n_movies]))
        return list(recommended_movies[:n_movies])