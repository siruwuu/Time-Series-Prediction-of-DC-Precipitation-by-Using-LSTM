import torch
import numpy as np
from torch.nn import Module, Linear, LSTM
from torch.utils.data import DataLoader, TensorDataset


class LSTMNet(Module):
    def __init__(self, config):
        super(LSTMNet, self).__init__()
        self.config = config

        self.lstm = LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_layers,
            batch_first=True,
            dropout=self.config.dropout_rate,
        )

        self.linear = Linear(
            in_features=self.config.lstm_hidden_size,
            out_features=self.config.ouput_size,
        )

    def forward(self, x, hidden=None):
        # LSTM forward computation
        lstm_out, hidden = self.lstm(x, hidden)
        # Get the dimension of LSTM output
        batch_size, time_step, hidden_size = lstm_out.shape

        if self.config.op == 0:
            # Reshape lstm_out to (batch_size*time_step, hidden_size) to pass into the fully connected layer
            lstm_out = lstm_out.reshape(-1, hidden_size)
            # Fully connected layer
            linear_out = self.linear(lstm_out)
            # Reshape for output
            output = linear_out.reshape(time_step, batch_size, -1)
            # Return only the last timestep data
            return output[-1]

        elif self.config.op == 1:
            linear_out = self.linear(lstm_out)  # Pass all timesteps through the fully connected layer
            output = linear_out[:, -1, :]  # Get the output of the last timestep
            return output

        elif self.config.op == 2:
            linear_out = self.linear(lstm_out[:, -1, :])  # Use the output of the last timestep of LSTM for the fully connected layer
            output = linear_out
            return output


def train(config, train_and_valid_data):
    model = LSTMNet(config).to(config.device)  # Instantiate the model

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    # Convert to Tensor
    train_X, train_Y, valid_X, valid_Y = (
        torch.from_numpy(train_X).float(),
        torch.from_numpy(train_Y).float(),
        torch.from_numpy(valid_X).float(),
        torch.from_numpy(valid_Y).float(),
    )

    # Use DataLoader to automatically generate trainable batch data
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Loss function
    criterion = torch.nn.MSELoss()

    valid_loss_min = float("inf")
    bad_epoch = 0
    train_loss = []
    valid_loss = []
    for epoch in range(config.epoch):
        model.train()  # Switch to training mode
        train_loss_array = []  # Record the loss of each batch
        hidden_train = None
        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(config.device), train_y.to(config.device)
            optimizer.zero_grad()  # Clear gradient information before training
            pred_y = model(train_x, hidden_train)  # Forward computation

            loss = criterion(pred_y, train_y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            train_loss_array.append(loss.item())

        # Early stopping mechanism: if the validation set prediction performance does not improve for config.patience epochs, stop to prevent overfitting
        model.eval()  # Switch to evaluation mode
        valid_loss_array = []  # Record the loss of each batch
        hidden_valid = None
        for valid_x, valid_y in valid_loader:
            valid_x, valid_y = valid_x.to(config.device), valid_y.to(config.device)
            pred_y = model(valid_x, hidden_valid)  # Forward computation

            loss = criterion(pred_y, valid_y)  # Compute loss

            valid_loss_array.append(loss.item())

        # Compute the total loss for this epoch
        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)

        # Print losses
        print(f"Epoch {epoch}/{config.epoch}: Train Loss is {train_loss_cur:.6f}, Valid Loss is {valid_loss_cur:.6f}")
        train_loss.append(train_loss_cur)
        valid_loss.append(valid_loss_cur)
        
        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path)  # Save model
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:  # Stop training if validation metric does not improve for patience epochs
                print(f"The training stops early in epoch {epoch}")
                break
            
    # Load the best model
    model.load_state_dict(torch.load(config.model_save_path))
    return model, train_loss, valid_loss


def predict(config, test_X, model_trained: LSTMNet = None):
    if model_trained is None:
        model_trained = LSTMNet(config).to(config.device)  # Instantiate the model
        state_dict = torch.load(config.model_load_path, weights_only=True)  # Load trained model parameters
        model_trained.load_state_dict(state_dict)

    # Convert to Tensor
    test_X = torch.from_numpy(test_X).float()
    # Use DataLoader to automatically generate batch data for testing
    test_loader = DataLoader(TensorDataset(test_X), batch_size=1)

    # Define a tensor to store predictions
    result = torch.Tensor().to(config.device)

    # Prediction process
    model_trained.eval()  # Switch to evaluation mode
    hidden_predict = None
    for test_x in test_loader:
        test_x = test_x[0].to(config.device)
        pred_y = model_trained(test_x, hidden_predict)
        cur_pred = torch.squeeze(pred_y, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()  # Remove gradient information, move to CPU if on GPU, and return as numpy array
