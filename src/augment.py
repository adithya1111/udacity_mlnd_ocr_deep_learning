import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import random
from scipy import ndarray

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}
def augment_image(img):
    #sk.io.imread('/home/CONCURASP/kumara/udacity_ocr/data/ramdisk/max/90kDICT32px/2425/1/98_Brawl_9292.jpg')
# random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))

    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](img)
        num_transformations += 1
    return transformed_image
#img_transform = augment_image(img)
#show_image(img_transform)