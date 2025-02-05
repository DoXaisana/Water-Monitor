import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from database import get_historical_data

# Load the trained model
class WaterUsageLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(WaterUsageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load model
model = WaterUsageLSTM()
model.load_state_dict(torch.load("machine-learning/Model/water_usage_model.pth"))
model.eval()

def predict_next_month():
    """Use AI model to predict water usage for the next 30 days."""
    df = get_historical_data()

    if len(df) < 30:
        return {"error": "Not enough data (needs at least 30 days)."}

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_usage"] = scaler.fit_transform(df[["daily_usage"]])
    
    # Prepare input sequence
    input_seq = df["scaled_usage"].values[-30:]

    # Predict next 30 days
    predictions = []
    for _ in range(30):
        with torch.no_grad():
            pred = model(torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1))
            pred_value = pred.item()
            predictions.append(pred_value)
            input_seq = np.roll(input_seq, -1)
            input_seq[-1] = pred_value

    # Convert back to actual scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    return {
        "dates": pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=30).strftime("%Y-%m-%d").tolist(),
        "predicted_usage": predictions.tolist(),
        "total_predicted": sum(predictions)
    }