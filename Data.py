import numpy as np
import struct

class Data:
    def __init__(self):
        pass

    def load_data(self, train_images_path, train_labels_path, test_images_path, test_labels_path):

        train_images, test_images = self.load_images(train_images_path, test_images_path)
        train_labels, test_labels = self.load_labels(train_labels_path, test_labels_path)
        return train_images, train_labels, test_images, test_labels

    def load_images(self, train_images_path, test_images_path):
        try:

            train_images = self.read_images(train_images_path)
            test_images = self.read_images(test_images_path)
            return train_images, test_images
                
        except Exception as ex:
            raise ex
        
    def load_labels(self, train_labels_path, test_label_path):
        try:

            train_labels = self.read_labels(train_labels_path)
            test_labels = self.read_labels(test_label_path)
            return train_labels, test_labels

        except Exception as ex:
            raise ex
        
    def read_images(self, images_path):
        try:

            with open(images_path, 'rb') as f:
                # Read the header information
                magic_number, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
                
                # Read the image data
                image_data = np.frombuffer(f.read(), dtype=np.uint8)
                images = image_data.reshape(num_images, rows, cols)
            return images

        except Exception as ex:
            raise ex
    
    def read_labels(self, labels_path):
        try:
            
            # Load train labels
            with open(labels_path, 'rb') as f:
                # Read the header information
                magic_number, num_labels = struct.unpack('>II', f.read(8))

                # Read the label data
                labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        
        except Exception as ex:
            raise ex
            