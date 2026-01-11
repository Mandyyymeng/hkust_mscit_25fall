import pandas as pd
import random


def check_test_data():
    ''' check generation'''
    df = pd.read_csv('test.csv', header=None, names=['number', 'label'])
    errors = []

    for idx, row in df.iterrows():
        num = str(row['number'])
        label = row['label']
        # check leading zeros
        if len(num) > 1 and num[0] == '0':
            errors.append(f"line {idx}: number {num} has leading zero")
            continue
        # check label
        is_pal = is_palindrome(num)
        if is_pal != bool(label):
            errors.append(f"line {idx}: number {num} wrong label - truth: {is_pal}, label: {label}")

    # print result
    if errors:
        print("ERROR:")
        for error in errors:
            print(error)
    else:
        print("checked correct")

def generate_palindrome(num_len):
    '''method for generating a palindrome number with length num_len'''
    # if odd number, has a single mid num; else not
    if num_len == 1:
        return random.randint(0, 9)
    first_digit = str(random.randint(1, 9))  # first digit should not be zero
    rest_digits = ''.join(str(random.randint(0, 9)) for _ in range(num_len//2 - 1)) if num_len > 2 else ''
    first_half = first_digit + rest_digits
    middle = str(random.randint(0, 9)) if num_len % 2 else ""
    palindrome = first_half + middle + first_half[::-1]

    return int(palindrome)

def generate_valid_number(num_len):
    ''' generate a valid number '''
    if num_len == 1:
        return str(random.randint(0, 9))
    else:
        return str(random.randint(10**(num_len-1), 10**num_len - 1))

def is_palindrome(num_str):
    return num_str == num_str[::-1]

def generate_test_data(num_samples=100,file_name = 'test.csv'):
    ''' generating test data '''
    samples = []
    labels = []
    num_neg = 0 # number of samples with label = 0 (not palindrome)
    num_pos = 0 # number of samples with label = 1 (palindrome)

    while num_neg < num_samples//2 and num_pos < num_samples//2:
        num_len = random.choices(range(1, 8), weights=[1, 1, 2, 3, 4, 5, 9])[0] #random.randint(1, 7)
        num = generate_valid_number(num_len)
        label = int(is_palindrome(num))
        num_neg += 1 - label
        num_pos += label
        labels.append(label)
        samples.append(int(num))

    samples_left = num_samples - len(samples)
    if num_neg >= num_samples//2:
        print("Enough negative, others all positive")
        for i in range(samples_left):
            num_len = random.choices(range(1, 8), weights=[2, 3, 3, 4, 5, 5, 8])[0]
            num = generate_palindrome(num_len)
            samples.append(num)
            labels.append(1)
    else:
        print("Enough positive, others all negative")
        for i in range(samples_left):
            num_len = random.randint(2, 7)
            num = generate_valid_number(num_len)
            while is_palindrome(num):
                num_len = random.randint(2, 7)
                num = generate_valid_number(num_len)
            samples.append(int(num))
            labels.append(0)

    # create and save DataFrame
    df = pd.DataFrame({'number': samples, 'label': labels})
    # random shuffle (frac=1: return a random ranked seq of all data)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(file_name, index=False, header=False)
    print(f"Generated test data with {len(samples)} samples")


if __name__ == "__main__":
    generate_test_data(50000,'train.csv')
    # check_test_data()
