from sklearn.model_selection import train_test_split
from model.model_pytorch import train, predict
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from codecarbon import EmissionsTracker
import os

class Config:
    def __init__(self):
        self.feature_columns = list(range(1, 5))  # Columns to be used as features, calculated starting from 0 in the original data. You can also use a list, such as [2,4,6,8].
        self.label_columns = [1]  # Columns to predict, calculated starting from 0 in the original data. For example, to predict both the 4th and 5th columns, use [4,5].
        self.label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(self.feature_columns, self.label_columns)  # Because features might not start from 0.

        # Network parameters
        self.input_size = len(self.feature_columns)
        self.ouput_size = len(self.label_columns)

        self.lstm_hidden_size = 256  # Hidden layer dimension of the LSTM
        self.lstm_layers = 2  # Number of stacked LSTM layers
        self.dropout_rate = 0.3  # Dropout probability
        self.time_step = 5  # This parameter is crucial: it sets how many past days of data to use for prediction, which is also the time step for the LSTM. Ensure training data is larger than this.

        # Path parameters
        self.train_data_path = "./data/processed_data.csv"

        # Path to save the trained model
        self.model_save_path = "./trained_model/model_lstm.pth"

        # Path to load the trained model
        self.model_load_path = "./trained_model/model_lstm.pth"

        # Path to save the prediction and test result data CSV
        self.csv_file_path = "./result_csv/predict_result.csv"

        # Path to save the final prediction effect plot
        self.predict_effect_path = "./fig/"

        # Division of training, testing, and validation datasets
        self.train_data_rate = 0.8  # Proportion of training data in the total dataset; the test data will be 1 - train_data_rate.
        self.valid_data_rate = 0.15  # Proportion of validation set used during training to choose models and parameters.

        # Training parameters
        self.batch_size = 128
        self.learning_rate = 0.0005
        self.epoch = 200  # The entire training set will be trained this many times, assuming no early stopping.
        self.patience = 10  # Stop training if the validation set does not improve for this many epochs.
        self.random_seed = 42  # Random seed to ensure reproducibility
        self.shuffle_train_data = True  # Whether to shuffle the training data

        # Control flags for training and prediction
        self.do_train = True
        self.do_predict = True
        self.use_trained_model = True

        # Choice of connection between LSTM and fully connected layers
        self.op = 2

        # Decides whether to use CUDA
        self.use_cuda = True
        self.device = torch.device("cuda:0" if self.use_cuda and torch.cuda.is_available() else "cpu")  # CPU or GPU training

# Save carbon emissions data to a CSV file
def save_emissions_to_csv(emissions):
    emission_data = {
        "project_name": ["LSTM-Precipitation"],
        "emissions_kg": [emissions],
    }
    df = pd.DataFrame(emission_data)
    df.to_csv("./result_csv/carbon_emissions.csv", index=False)
    print("Carbon emissions saved to carbon_emissions.csv")

# Main function

class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()  # Read data from the CSV file

        self.data_num = self.data.shape[0]  # Total number of data points
        self.train_num = int(self.data_num * self.config.train_data_rate)  # Number of training data points
        self.test_num = self.data_num - self.train_num  # Number of test data points

        self.scalerX = preprocessing.MinMaxScaler()  # Used for normalizing input features
        self.scalerY = preprocessing.MinMaxScaler()  # Used for normalizing output

    def read_data(self):
        init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()  # columns.tolist() gets column names

    def get_train_and_valid_data(self):
        # Obtain the training set
        feature_data = self.data[: self.train_num]
        # Use data delayed by time_step rows as labels
        label_data = self.data[self.config.time_step : self.config.time_step + self.train_num, self.config.label_in_feature_index]

        # Each time_step rows will be one sample, and two samples are staggered by one row, e.g., rows 1-20 and 2-21
        train_X = [feature_data[i : i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
        train_Y = [label_data[i] for i in range(self.train_num - self.config.time_step)]

        # Split the training set into a validation set, and shuffle
        train_X, valid_X, train_Y, valid_Y = train_test_split(
            train_X,
            train_Y,
            test_size=self.config.valid_data_rate,
            random_state=self.config.random_seed,
            shuffle=self.config.shuffle_train_data,
        )

        # Convert to ndarray
        train_X, valid_X, train_Y, valid_Y = np.array(train_X), np.array(valid_X), np.array(train_Y), np.array(valid_Y)

        # Reshape for normalization, since the required dimension for normalization must be less than 2
        train_X, valid_X = train_X.reshape((train_X.shape[0], -1)), valid_X.reshape((valid_X.shape[0], -1))

        # Normalize the training set
        norm_train_X = self.scalerX.fit_transform(train_X)
        norm_train_Y = self.scalerY.fit_transform(train_Y)

        # Normalize the validation set
        norm_valid_X = self.scalerX.transform(valid_X)
        norm_valid_Y = self.scalerY.transform(valid_Y)

        # Reshape back
        norm_train_X = norm_train_X.reshape((norm_train_X.shape[0], -1, self.config.input_size))
        norm_valid_X = norm_valid_X.reshape((norm_valid_X.shape[0], -1, self.config.input_size))

        return norm_train_X, norm_train_Y, norm_valid_X, norm_valid_Y

    def get_test_data(self, return_label_data=False):
        # Obtain the test set
        feature_data = self.data[self.train_num :]
        # Use data delayed by time_step rows as labels
        label_data = self.data[self.config.time_step + self.train_num :, self.config.label_in_feature_index]

        # Each time_step rows will be one sample, and two samples are staggered by one row, e.g., rows 1-20 and 2-21
        test_X = [feature_data[i : i + self.config.time_step] for i in range(self.test_num - self.config.time_step)]
        test_Y = [label_data[i] for i in range(self.test_num - self.config.time_step)]

        # Convert to ndarray
        test_X, test_Y = np.array(test_X), np.array(test_Y)

        # Reshape for normalization, since the required dimension for normalization must be less than 2
        test_X = test_X.reshape((test_X.shape[0], -1))

        # Normalize the test set
        norm_test_X = self.scalerX.transform(test_X)
        norm_test_Y = self.scalerY.transform(test_Y)

        # Reshape back
        norm_test_X = norm_test_X.reshape((norm_test_X.shape[0], -1, self.config.input_size))

        if return_label_data:  # In real-world applications, test sets do not include label data
            return norm_test_X, norm_test_Y
        return norm_test_X


"""Plotting"""

# def plot(config, test_Y, pred_Y_mean):
#     f, ax = plt.subplots(1, 1)
#     x_axis = np.arange(test_Y.shape[0])
#     ax.plot(x_axis, test_Y, label="real value", c="r")  # Plot real values as a red line
#     ax.plot(x_axis, pred_Y_mean, label="predicted value", c="b")  # Plot predicted mean values as a blue line
#     ax.grid(linestyle="--", alpha=0.5)  # Draw gridlines
#     ax.legend()  # Draw legend
#     # Title
#     ax.set_title("Prediction Result")
#     # Axes labels
#     ax.set_xlabel("24 hours/sample")
#     ax.set_ylabel("meters")
#     plt.savefig(config.predict_effect_path + "Forecasting hidden size 256 epoch 200.jpg")

def plot(config, test_Y, pred_Y_mean):
    f, ax = plt.subplots(1, 1)
    
    # Update x-axis to reflect the correct time resolution
    x_axis = np.arange(test_Y.shape[0]) * 12  # Multiply by 12 to convert sample index to hours
    ax.plot(x_axis, test_Y, label="Real Value", color="red")  # Use clear labels with capitalized terms
    ax.plot(x_axis, pred_Y_mean, label="Predicted Value", color="blue")  # Use clear labels with capitalized terms
    
    # Add grid and legend
    ax.grid(linestyle="--", alpha=0.5)  # Draw gridlines
    ax.legend(loc="upper right")  # Move legend to a clear position
    
    # Update title and axes labels for clarity
    ax.set_title("Prediction Results", fontsize=14)  # Update title for clarity
    ax.set_xlabel("Time (hours)", fontsize=12)  # Clarify x-axis units
    ax.set_ylabel("Precipitation (meters)", fontsize=12)  # Clarify y-axis units
    
    # Improve overall appearance of plot
    plt.xticks(fontsize=10)  # Adjust font size for x-axis ticks
    plt.yticks(fontsize=10)  # Adjust font size for y-axis ticks
    plt.tight_layout()  # Ensure everything fits without overlap

    # Save the plot
    plt.savefig(config.predict_effect_path + "Prediction_result_12_hour_resolution.jpg", dpi=300)


def plot_loss(config, train_loss, valid_loss):
    if not train_loss or not valid_loss:  # Check if empty
        print("Train Loss or Validation Loss is empty. Skipping plot.")
        return

    plt.figure()
    x_axis = [k + 1 for k in range(len(train_loss))]  # Horizontal axis
    plt.plot(x_axis, train_loss, label="train_loss", c="r")  # Plot training loss as a red line
    plt.plot(x_axis, valid_loss, label="valid_loss", c="b")  # Plot validation loss as a blue line

    # Set y-axis range with an additional margin
    y_min = min(min(train_loss), min(valid_loss))
    y_max = max(max(train_loss), max(valid_loss))
    y_margin = (y_max - y_min) * 0.1  # Add 10% margin
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.grid(linestyle="--", alpha=0.5)  # Draw gridlines
    plt.legend()  # Draw legend

    # Title
    title_string = "LOSS when hidden size 256 epoch 200"
    plt.title(title_string)

    # Axes labels
    plt.xlabel("Epochs")
    plt.ylabel("Meters")

    # Ensure the save path exists
    if not os.path.exists(config.predict_effect_path):
        os.makedirs(config.predict_effect_path)

    # Save the plot and display it
    plt.savefig(config.predict_effect_path + title_string + ".jpg", dpi=300)
    plt.show()


"""Save to CSV file"""


def save_to_csv(config, test_Y, pred_Y_mean):
    task = config.ouput_size
    for i in range(task):
        data = {
            "test_Y": test_Y[:, i].tolist(),
            "pred_Y": pred_Y_mean[:, i].tolist(),
        }
        df = pd.DataFrame(data)
        df.to_csv(config.csv_file_path, index=False)


"""Calculate fitting metrics"""


def measurement(test_Y, pred_Y_mean):
    mae = np.mean(np.abs(pred_Y_mean - test_Y))
    rmse = np.mean(np.power(pred_Y_mean - test_Y, 2))
    mape = np.mean(np.abs((pred_Y_mean - test_Y) / test_Y)) * 100
    r2 = 1 - np.sum((test_Y - pred_Y_mean) ** 2) / np.sum((test_Y - np.mean(test_Y)) ** 2)
    print(f"Model regression evaluation indicators :\nMAE:{mae:.6f}\nRMSE:{rmse:.6f}\nMAPE:{mape:.6f}%\nR2:{r2:.6f}")


def main(config: Config):
    np.random.seed(config.random_seed)  # Set random seed for reproducibility
    data_gainer = Data(config)

    train_X, train_Y, valid_X, valid_Y = data_gainer.get_train_and_valid_data()
    test_X, test_Y = data_gainer.get_test_data(return_label_data=True)

    if config.do_train:  # If training is enabled
        # Start CodeCarbon tracker
        tracker = EmissionsTracker(output_dir="./result_csv", project_name="LSTM-Precipitation_Carbon")
        tracker.start()

        model, train_loss, valid_loss = train(config, [train_X, train_Y, valid_X, valid_Y])

        # Stop emissions tracking
        emissions = tracker.stop()
        save_emissions_to_csv(emissions)

        start_index = 0  # Start plotting from the first epoch
        plot_loss(config, train_loss[start_index:], valid_loss[start_index:])

    if config.do_predict:  # If prediction is enabled
        print("Starting prediction...")
        
        # Ensure the trained model is loaded successfully, even if saved via early stopping
        if config.use_trained_model:
            model = None  # Force loading of the trained model
        pred_Y = predict(config, test_X, model)
        print("Prediction completed.")

        # Inverse transform results
        test_Y = data_gainer.scalerY.inverse_transform(test_Y.reshape(-1, 1))
        pred_Y = data_gainer.scalerY.inverse_transform(pred_Y.reshape(-1, 1))

        # Plot results
        plot(config, test_Y, pred_Y)

        # Compute evaluation metrics
        measurement(test_Y, pred_Y)

        # Save results to CSV
        save_to_csv(config, test_Y, pred_Y)
        print("Prediction and CSV save completed.")

if __name__ == "__main__":
    # Main execution entry
    con = Config()
    main(con)
