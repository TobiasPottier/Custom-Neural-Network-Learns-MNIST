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


X_train_orig, y_train, X_test_orig, y_test = load_data()
num_classes = 10
y_train = np.eye(num_classes)[y_train].T
y_test = np.eye(num_classes)[y_test].T
print(f'Number of train images: {X_train_orig.shape[0]}')
print(f'Number of test images: {X_test_orig.shape[0]}')
print(f'Image Resolution: {X_train_orig.shape[1]} x {X_train_orig.shape[2]}')
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

print ("\ntrain_x's shape: " + str(X_train.shape))
print ("test_x's shape: " + str(X_test.shape))

model = NeuralNetwork()

layers_dims = [X_train.shape[0], 256, 128, num_classes]


print('Training...')
parameters, costs = model.train_model(X_train, y_train, layers_dims, num_iterations = 4000, print_cost = True, print_cost_every_iterations = 100)
print(f'\n\n{"-"*50}')
print(costs)
print(f'\n\n{"-"*50}')

import json
# Convert numpy arrays to lists for JSON serialization
parameters_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in parameters.items()}

# Save to JSON file
with open('parameters.json', 'w') as json_file:
    json.dump(parameters_serializable, json_file)
print("Parameters saved to 'parameters.json'")


print(f'\n\n{"-"*50}')
print(f'{"-"*50}')
print('Parameters Saved!')
print(f'{"-"*50}')
print(f'{"-"*50}\n\n')


user_input = input('ready> ')

i = 5
x_single = X_test_flatten[:, i].reshape(784, 1)
prediction, confidences = model.predict(x_single, parameters)
print(prediction)
print(confidences)

plt.imshow(X_test_orig[i], cmap='grey')
plt.show()