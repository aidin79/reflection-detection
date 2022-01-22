from helpers import load_images, reflection_detection

images = load_images('./dataset/')

for i in range(5):
    reflection_detection(images[2 * i], images[2 * i + 1])
