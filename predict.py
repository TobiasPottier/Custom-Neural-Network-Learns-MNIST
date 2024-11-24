from NeuralNetwork import NeuralNetwork
from Data import Data
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    train_images_path = './dataset/train-images.idx3-ubyte'
    train_labels_path = './dataset/train-labels.idx1-ubyte'
    test_images_path = './dataset/t10k-images.idx3-ubyte'
    test_labels_path = './dataset/t10k-labels.idx1-ubyte'

    return Data().load_data(train_images_path, train_labels_path, test_images_path, test_labels_path)


X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_data()
num_classes = 10
y_test = np.eye(num_classes)[y_test_orig].T
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_test = X_test_flatten/255.

model = NeuralNetwork()

print('Pulling model...')

import json
# Load parameters from JSON file
with open('parameters.json', 'r') as json_file:
    loaded_parameters = json.load(json_file)

# Convert lists back to numpy arrays
parameters = {k: np.array(v) if isinstance(v, list) else v for k, v in loaded_parameters.items()}

print(f'\n\n{"-"*50}')
print(f'{"-"*50}')
print('Model Complete!')
print(f'{"-"*50}')
print(f'{"-"*50}\n\n')

user_input = input('ready> ')

i = 30
x_single = X_test[:, i].reshape(784, 1)
prediction, confidences = model.predict(x_single, parameters)
print(prediction)
print(confidences)

plt.imshow(X_test_orig[i], cmap='grey')
plt.show()