from PIL import Image, ImageDraw, ImageFont
import math

def transform_vector_to_class(vector, label_names, threshold):
    """
    This function use for transform output vector to list of labels
    """
    # Initialize an empty list to store selected labels
    selected_labels = []
    
    # Iterate through each element in the vector and its corresponding label
    for value, label in zip(vector, label_names):
        if value > threshold:
            selected_labels.append(label)
    
    return selected_labels

def add_labels_to_image(image_path, selected_labels, save_path):
    """
    This function use for caption an image by it labels
    """
    # Load the image
    image = Image.open(image_path)

    # Define font and size for the labels
    font = ImageFont.load_default()
    font_size = 20

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Calculate the position to start displaying labels
    position = (10, 10)

    # Iterate through the selected labels and draw them on the image
    for label in selected_labels:
        draw.text(position, label, font=font, fill=(255, 255, 255))
        position = (position[0], position[1] + font_size + 5)  # Move down for the next label

    # Save or display the modified image
    # image.show()  # Display the image
    image.save(save_path)  # Save the image with labels

def add_labels_to_images(image_paths, selected_labels_list, output_path, cols=3):
    """
    This function is use for create a grid of images with its labels
    """
    # Define font and size for the labels
    font = ImageFont.load_default()
    font_size = 20

    # Calculate the maximum label width for positioning
    max_label_width = max([font.getsize(label)[0] for labels in selected_labels_list for label in labels])

    # Load and process each image
    images_with_labels = []
    for image_path, selected_labels in zip(image_paths, selected_labels_list):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Start position for labels
        position = (10, 10)

        # Draw selected labels on the image
        for label in selected_labels:
            draw.text(position, label, font=font, fill=(255, 255, 0))
            position = (position[0], position[1] + font_size + 5)  # Move down for the next label

        images_with_labels.append(image)

    # Calculate grid size for arranging images
    num_images = len(images_with_labels)
    rows = math.ceil(num_images / cols)  # Calculate number of rows

    # Create a blank canvas for the grid
    canvas_width = cols * max([image.width for image in images_with_labels])
    canvas_height = rows * max([image.height for image in images_with_labels])
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))

    # Arrange images in the grid
    for i, image in enumerate(images_with_labels):
        col = i % cols
        row = i // cols
        canvas.paste(image, (col * image.width, row * image.height))

    #canvas.show()  # Display the grid of images with labels
    canvas.save(output_path)  # Save the grid image with labels