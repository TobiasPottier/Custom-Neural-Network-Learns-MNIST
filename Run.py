import threading
from DrawingBoard import DrawingBoard
import tkinter as tk
from NeuralNetwork import NeuralNetwork
import numpy as np
import json

# Background function
def model_prediction():
    while True:
        array = drawing_board.get_array()
        array = array.reshape(28 * 28, 1)
        prediction, confidence = model.predict(array, parameters)
        drawing_board.update_text(prediction.squeeze(), confidence.squeeze())


model = NeuralNetwork()

print('Pulling model...')
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

root = tk.Tk()
root.title(">>AI Drawing Board<<")
drawing_board = DrawingBoard(root)

# Start the model_prediction in a separate thread
thread = threading.Thread(target=model_prediction, daemon=True)
thread.start()

root.mainloop()