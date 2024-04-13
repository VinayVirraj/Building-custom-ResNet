import numpy as np
from matplotlib import pyplot as plt


"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])
    ### END CODE HERE

    image = preprocess_image(image, training) # If any.
    image = np.transpose(image, [2, 0, 1])

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        height = image.shape[0]
        width = image.shape[1]
        resized_img = np.zeros((height + 2 * 4, width + 2 * 4, 3))
        resized_img[4:4 + height, 4:4 + width] = image
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        x = np.random.randint(0, resized_img.shape[0] - 32)
        y = np.random.randint(0, resized_img.shape[1] - 32)
        image = resized_img[y:y+32, x:x+32]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        flip = np.random.randint(2)
        if flip == 1:
            image = np.fliplr(image)
        
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    image = (image - np.mean(image))/np.std(image)
    ### YOUR CODE HERE
    return image
    ### END CODE HERE


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

### END CODE HERE