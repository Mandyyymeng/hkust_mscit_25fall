import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import argparse
import os
import time
import math
import random


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=10):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # add an embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.i2h = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        print(f"Model Initialized with number of parameters: "
              f"{sum(p.numel() for p in self.parameters())}")

    def forward(self, input, hidden):
        # input shape: (batch_size, 1)
        embedded = self.embedding(input)  # (batch_size, 1, embedding_dim)
        embedded = embedded.squeeze(1)  # (batch_size, embedding_dim)

        combined = torch.cat((embedded, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)


def load_training_data(csv_file='train.csv', num_samples=None):
    '''load data from csv file'''
    df = pd.read_csv(csv_file, header=None, names=['number', 'label'])
    if num_samples is not None:
        df = df.head(num_samples)  # read the first num_samples lines
    samples = df['number'].astype(str).tolist()
    labels = df['label'].tolist()
    return samples, labels

def numbers_to_tensor(number_strs, max_length=7):
    """Convert list of number strings to batch tensor"""
    batch_tensors = []
    for number_str in number_strs:
        digits = [int(d) for d in number_str]
        while len(digits) < max_length:
            digits.insert(0, 0)  # padding
        batch_tensors.append(digits)

    return torch.tensor(batch_tensors, dtype=torch.long)  # (batch_size, max_length)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def plot_training_curves(losses, accuracies, num_samples, save_path=None):
    """Plot training loss and accuracy curves"""
    plt.figure(figsize=(10, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.title(f'Training Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, 'r-', linewidth=2, label='Training Accuracy')
    plt.title(f'Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {save_path}")
    else:
        plt.show()

    plt.close()

def train_model(num_train_samples, model_name, num_epochs=50, hidden_size=128, learning_rate=0.001,
                use_optimizer=False, batch_size=32):
    """
    Training function with CUDA support and batch processing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using optimizer: {use_optimizer}")
    print(f"Batch size: {batch_size}")

    train_samples, train_labels = load_training_data(num_samples=num_train_samples)

    input_size = 11
    output_size = 2
    embedding_dim = 10

    model = RNN(input_size, hidden_size, output_size, embedding_dim)
    model.to(device)

    if use_optimizer:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    all_losses = []
    all_accuracies = []
    print_every = max(1, num_epochs // 10)
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        correct = 0

        # Shuffle data each epoch
        indices = list(range(len(train_samples)))
        random.shuffle(indices)
        shuffled_samples = [train_samples[i] for i in indices]
        shuffled_labels = [train_labels[i] for i in indices]

        # Process by batch
        for batch_start in range(0, len(shuffled_samples), batch_size):
            batch_end = batch_start + batch_size
            batch_samples = shuffled_samples[batch_start:batch_end]
            batch_labels = shuffled_labels[batch_start:batch_end]
            actual_batch_size = len(batch_samples)

            # Convert to batch tensors
            sequence_batch = numbers_to_tensor(batch_samples).to(device)  # (batch_size, max_length)
            target_batch = torch.tensor(batch_labels, dtype=torch.long).to(device)  # (batch_size,)

            # Initialize hidden state for this batch
            hidden = model.initHidden(batch_size=actual_batch_size).to(device)

            if use_optimizer:
                optimizer.zero_grad()
            else:
                model.zero_grad()

            # Process sequence
            for i in range(sequence_batch.size(1)):  # iterate through sequence length
                input_tensor = sequence_batch[:, i].unsqueeze(1)  # (batch_size, 1)
                output, hidden = model(input_tensor, hidden)

            loss = criterion(output, target_batch)
            loss.backward()

            if use_optimizer:
                optimizer.step()
            else:
                for p in model.parameters():
                    p.data.add_(p.grad.data, alpha=-learning_rate)

            total_loss += loss.item() * actual_batch_size
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == target_batch).sum().item()

        epoch_loss = total_loss / len(train_samples)
        accuracy = correct / len(train_samples)
        all_losses.append(epoch_loss)
        all_accuracies.append(accuracy)

        if epoch % print_every == 0:
            print('%d %d%% (%s) loss: %.4f acc: %.4f' % (
                epoch, epoch / num_epochs * 100, timeSince(start), epoch_loss, accuracy))

    torch.save(model.state_dict(), f'{model_name}.pth')
    plot_training_curves(all_losses, all_accuracies, num_train_samples, f'training_curves_{num_train_samples}.png')
    print(f"Final accuracy: {accuracy:.4f}")
    return model, all_losses, all_accuracies

def all_models_test(model_sizes, test_file='test.csv'):
    """ test multiple models """
    input_size = 11
    output_size = 2
    hidden_size = 192 #128
    embedding_dim = 15

    # load test data
    test_data = pd.read_csv(test_file, header=None)
    test_samples = test_data[0].astype(str).tolist()
    test_labels = test_data[1].tolist()

    print("Testing on:", test_file)
    print("-" * 60)

    for num_samples in model_sizes:
        model_name = f'palindrome_rnn_{num_samples}.pth'
        if not os.path.exists(model_name):
            print(f"Model {model_name} not found, please train first.")
            continue

        model = RNN(input_size, hidden_size, output_size, embedding_dim)
        model.load_state_dict(torch.load(model_name, weights_only=True))
        model.eval()

        predictions = []
        for sample in test_samples:
            hidden = model.initHidden()
            sequence = numbers_to_tensor(sample)
            for i in range(len(sequence)):
                input_tensor = sequence[i].unsqueeze(0).unsqueeze(0)
                output, hidden = model(input_tensor, hidden)
            predicted = torch.argmax(output, dim=1).item()
            predictions.append(predicted)

        accuracy = accuracy_score(test_labels, predictions)
        print(f"Model trained on {num_samples} samples: Accuracy = {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Palindrome RNN Training and Testing')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--model', type=str, default="200", help='Model sizes to use for testing("200,1000",50000)')
    parser.add_argument('--use_optimizer', action='store_true', help='Use optimizer instead of manual updates')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    args = parser.parse_args()
    model_sizes = [int(x.strip()) for x in args.model.split(',')]

    if args.train:
        sample_sizes = model_sizes
        print(f"STARTING TRAINING FOR {len(sample_sizes)} MODELS, Sample sizes: {sample_sizes}")

        for i, num_samples in enumerate(sample_sizes, 1):
            print(f"\n[{i}/{len(sample_sizes)}] ", end="")
            model_name = f'palindrome_rnn_{num_samples}'
            train_model(num_samples, model_name, num_epochs=350, learning_rate=0.0035,
                        use_optimizer=args.use_optimizer, batch_size=args.batch_size) #0.001

        print(f"ALL TRAINING COMPLETED")

    if args.test:
        all_models_test(model_sizes=model_sizes)

if __name__ == "__main__":
    main()

