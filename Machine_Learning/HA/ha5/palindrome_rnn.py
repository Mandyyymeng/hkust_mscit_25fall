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
        # add embedding
        embedded = self.embedding(input)
        if embedded.dim() > 2:
            embedded = embedded.squeeze(1)

        combined = torch.cat((embedded, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size=1):
        # modified 1 to "batch size"
        return torch.zeros(batch_size, self.hidden_size)

def load_training_data(csv_file='train.csv', num_samples=None):
    '''load data from csv file'''
    df = pd.read_csv(csv_file, header=None, names=['number', 'label'])
    if num_samples is not None:
        df = df.head(num_samples) # read the first num_samples lines
    samples = df['number'].astype(str).tolist()
    labels = df['label'].tolist()
    return samples, labels

def number_to_tensor(number_str, max_length=7):
    """ turn number to a tensor"""
    digits = [int(d) for d in number_str]
    while len(digits) < max_length:
        digits.insert(0, 10)  # 使用10作为填充符号
    tensor = torch.tensor(digits, dtype=torch.long)
    return tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def plot_training_curves(losses, accuracies, num_samples, save_path=None):
    """Plot training loss and accuracy curves"""
    plt.figure(figsize=(10,5))

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
                use_optimizer=False):
    """
    Training function with CUDA support
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Using optimizer: {use_optimizer}")

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

        for sample, label in zip(train_samples, train_labels):
            hidden = model.initHidden().to(device)
            sequence = number_to_tensor(sample).to(device)
            target = torch.tensor([label], dtype=torch.long).to(device)

            if use_optimizer:
                optimizer.zero_grad()
            else:
                model.zero_grad()

            for i in range(len(sequence)):
                input_tensor = sequence[i].unsqueeze(0).unsqueeze(0)
                output, hidden = model(input_tensor, hidden)

            loss = criterion(output, target)
            loss.backward()

            if use_optimizer:
                optimizer.step()  # 移除了梯度裁剪
            else:
                for p in model.parameters():
                    p.data.add_(p.grad.data, alpha=-learning_rate)

            total_loss += loss.item()
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == target).sum().item()

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


def all_models_test(model_sizes,test_file='test.csv'):
    """ test multiple models """
    input_size = 11
    output_size = 2
    hidden_size = 128
    embedding_dim = 10

    # load test data
    test_data = pd.read_csv(test_file, header=None)
    test_samples = test_data[0].astype(str).tolist()
    test_labels = test_data[1].tolist()

    print("Testing models on:", test_file)
    print("-" * 60)

    for num_samples in model_sizes:
        model_name = f'palindrome_rnn_{num_samples}.pth'
        if not os.path.exists(model_name):
            print(f"Model {model_name} not found, please train first.")
            continue

        model = RNN(input_size, hidden_size, output_size, embedding_dim)
        model.load_state_dict(torch.load(model_name, weights_only=True)) # need to specify
        model.eval()

        predictions = []
        for sample in test_samples:
            hidden = model.initHidden()
            sequence = number_to_tensor(sample)
            for i in range(len(sequence)):
                input_tensor = sequence[i].unsqueeze(0).unsqueeze(0)
                output, hidden = model(input_tensor, hidden)
            predicted = torch.argmax(output, dim=1).item()
            predictions.append(predicted)

        accuracy = accuracy_score(test_labels, predictions)
        print(f"Test on {len(test_samples)} samples on models {model_sizes}, accuracy = {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Palindrome RNN Training and Testing')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--model', type=str, default="200", help='Model sizes to use for testing("200,1000",50000)')
    parser.add_argument('--use_optimizer', action='store_true', help='Use optimizer instead of manual updates')

    args = parser.parse_args()
    model_sizes = [int(x.strip()) for x in args.model.split(',')]

    if args.train:
        sample_sizes = model_sizes
        print(f"STARTING TRAINING FOR {len(sample_sizes)} MODELS, Sample sizes: {sample_sizes}")

        for i, num_samples in enumerate(sample_sizes, 1):
            print(f"\n[{i}/{len(sample_sizes)}] ", end="")
            model_name = f'palindrome_rnn_{num_samples}'
            # train_model_optimizer(num_samples,model_name)
            train_model(num_samples, model_name, num_epochs=50, learning_rate=0.0012, use_optimizer=args.use_optimizer)

        print(f"ALL TRAINING COMPLETED")

    if args.test:
        all_models_test(model_sizes=model_sizes)


if __name__ == "__main__":
    main()

