import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Changed from MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam


class WaterUsagePrediction:
    def __init__(self, data_dir="water_usage_data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()  # Using StandardScaler to avoid NaN issues
        self.model = None
        self.sequence_length = 48 * 7  # One week of data

    def load_and_prepare_data(self):
        """Load all CSV files and prepare data for training."""
        all_files = glob.glob(os.path.join(self.data_dir, "water_usage_*.csv"))
        if not all_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        dfs = [pd.read_csv(filename) for filename in all_files]
        df = pd.concat(dfs, ignore_index=True)

        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.sort_values('DateTime')

        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['Month'] = df['DateTime'].dt.month
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

        df['IsSpecialEvent'] = df['Special Events'].apply(lambda x: 0 if x == 'Normal day' else 1)
        df['IsLaoNewYear'] = df['Special Events'].str.contains('Lao New Year').astype(int)

        features = ['Usage Percentage', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsSpecialEvent', 'IsLaoNewYear']
        self.data = df[features].dropna()

        # Check for invalid values
        if not np.all(np.isfinite(self.data.values)):
            raise ValueError("Training data contains NaN or infinite values!")

        print(f"Loaded {len(self.data)} valid data points")
        return self.data

    def create_sequences(self, data):
        """Create sequences for LSTM training with multiple features."""
        X, y = [], []
        values = data.values  

        if len(values) <= self.sequence_length:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length + 1}")

        for i in range(len(values) - self.sequence_length):
            sequence = values[i:(i + self.sequence_length)]
            target = values[i + self.sequence_length][0]  

            if np.isnan(sequence).any() or np.isnan(target):
                continue  # Skip sequences with NaNs

            X.append(sequence)
            y.append(target)

        if not X:
            raise ValueError("No valid sequences could be created")

        return np.array(X), np.array(y)

    def build_model(self):
        """Create LSTM model architecture."""
        model = Sequential([
            LSTM(64, activation='tanh', input_shape=(self.sequence_length, 7), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def train_model(self, epochs=20, batch_size=32):
        """Train the model with the prepared data."""
        print("Loading and preparing data...")
        data = self.load_and_prepare_data()

        scaled_data = self.scaler.fit_transform(data)  

        print("Creating sequences...")
        X, y = self.create_sequences(pd.DataFrame(scaled_data, columns=data.columns))

        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

        print(f"Created {len(X)} sequences for training")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Building and training model...")
        self.model = self.build_model()

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=1
        )

        self.save_model()  
        return history

    def predict_next_month(self, last_month_data):
        """Predict water usage for the next month."""
        if self.model is None:
            raise ValueError("Model needs to be trained first")

        # Ensure we have enough data points
        if len(last_month_data) < self.sequence_length + 1:
            print(f"Warning: Not enough data. Need at least {self.sequence_length + 1} points. Using available {len(last_month_data)} points instead.")
            last_month_data = self.data.tail(self.sequence_length + 1)  # Take from full dataset

        scaled_data = self.scaler.transform(last_month_data)

        X, _ = self.create_sequences(pd.DataFrame(scaled_data, columns=last_month_data.columns))

        if len(X) == 0:
            raise ValueError("No valid sequences created for prediction!")

        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

        scaled_prediction = self.model.predict(X)
        prediction = self.scaler.inverse_transform(scaled_prediction)

        return prediction.flatten()[-1]  # Return last predicted value

    def evaluate_model(self, test_data):
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model needs to be trained first")

        scaled_data = self.scaler.transform(test_data)

        X, y = self.create_sequences(pd.DataFrame(scaled_data, columns=test_data.columns))

        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

        scaled_predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(scaled_predictions)

        mse = mean_squared_error(y, scaled_predictions)
        mae = mean_absolute_error(y, scaled_predictions)
        r2 = r2_score(y, scaled_predictions)

        return {'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE': np.sqrt(mse)}

    def save_model(self, model_path="water_usage_model.h5"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        self.model.save(model_path)
        print(f"Model saved at {model_path}")

    def load_model(self, model_path="water_usage_model.h5"):
        """Load a previously trained model."""
        if not os.path.exists(model_path):
            raise ValueError(f"Model file '{model_path}' not found!")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")


if __name__ == "__main__":
    try:
        predictor = WaterUsagePrediction()

        print("Starting model training...")
        history = predictor.train_model(epochs=20)

        print("\nPreparing recent data for prediction...")
        recent_data = predictor.load_and_prepare_data().tail(48 * 7)  # Ensure at least 336 points

        print("\nMaking prediction...")
        next_month_usage = predictor.predict_next_month(recent_data)
        print(f"Predicted usage for next month: {next_month_usage:.2f}%")

        print("\nEvaluating model performance...")
        metrics = predictor.evaluate_model(predictor.data)
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")