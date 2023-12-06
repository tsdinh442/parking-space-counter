from PIL import Image
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.image import flip_left_right, adjust_brightness


def augment(image):
    '''
    given an image, return an augmneted version of the image
    '''
    image = flip_left_right(image)
    #image = adjust_brightness(image, delta=0.1)
    return image

def load_data(directory_path, image_size, augmentation=False):
    '''
    load the images from the folder; if augnmentation set to True, load both the original and the augmented images
    input: 
    directory_path: str, path to the folder containing the images
    image_size: tuple, the size the images will be resized to
    augmentation: bool, whether nor not to load the aumented images

    return:
    images, labels (matching the images)
    '''
    
    images = []
    labels = []
    for label in os.listdir(directory_path):
        for filename in os.listdir(os.path.join(directory_path, label)):
            # Get the full path of the image file
            file_path = os.path.join(directory_path, label, filename)

            # Check if the file is a regular file
            if os.path.isfile(file_path):
                # Extract the folder name from the file path
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    images.append(img)
                    labels.append(label)
                    if augmentation:
                        images.append(augment(img))
                        labels.append(label)

    return images, labels

def extract_parking_stalls(image_path, mask_path, folder_path):
    '''
    cropping parking stalls from an aeriel images of parking lots
    save the cropped images in the designated folder
    each image is named numerically starting from 0
    :param image_path: path to the image
    :param mask_path: path to the corresponding mask
    :param folder_path: path to the folder where the cropped images will be saved
    :return: None
    '''

    # Read the image
    image = cv2.imread(image_path)
    # Read the mask file (already black and white)
    mask = cv2.imread(mask_path, 0)

    # Find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour
    for i, contour in enumerate(contours):
        # Create an empty mask for the contour
        contour_mask = np.zeros_like(mask)

        # Draw the current contour on the mask
        cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

        # Bitwise AND operation to crop the image using the contour mask
        cropped = cv2.bitwise_and(image, cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR))

        # Find the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the region from the original image using the bounding rectangle
        cropped_mask = cropped[y:y + h, x:x + w]

        # Display or save the cropped image
        # cv2.imshow('Cropped Image', resized)
        # cv2.waitKey(0)

        # Save the cropped image
        cv2.imwrite(folder_path + '{}.jpg'.format(i), cropped_mask)
        return

def helper_predict(vgg, model, data, input_size):
    '''
    preparing the image data to the correct size and shape, extract the features, then classify the images
    input:
    vgg: pretrained vgg16 model
    mode: the trained model
    data: the image to be classified
    input_size: tuple, the size of the image that was trained on the model

    return: 
    pred: the predicted catagory of the image
    '''

    # resize and normalize
    data = cv2.resize(data, input_size)
    data = data / 255
    input_img = np.expand_dims(data, axis=0)

    # extract the features by running it thru the vgg16 model
    feature_extractor = vgg.predict(input_img)

    # reshape the feature to the correct shape
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)

    # classifying the image
    pred = model.predict(features)
    
    return pred

def parking_counter(video_path, mask_path, output_folder_path, input_size, vgg, model):
    '''
    predict the number of cars in a video
    input:
    video_path: str, path to the video
    mask_path: str, path to the mask of the frame
    output_folder_path: str, path to the folder where each predicted frame to be saved
    input_size: tuple, the size of the image that the model was trained on
    vgg: pretrained vgg 16
    mode: the model used for prediction

    return None
    '''
    
    # Define colors in BGR color space
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    # Define the text and its properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Read the mask file (already black and white)
    mask = cv2.imread(mask_path, 0)

    # Find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ret = True
    frame_count = 0
    while ret:

        ret, frame = cap.read()
        if frame_count > 47:  # start from frame 48 because the first few frames are transitional
            overlay = frame.copy()

            if frame_count % 12 == 0:
                empty = []
                occupied = []

                # Iterate over each contour
                for i, contour in enumerate(contours):
                    # Create an empty mask for the contour
                    contour_mask = np.zeros_like(mask)

                    # Draw the current contour on the mask
                    cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

                    # Bitwise AND operation to crop the image using the contour mask
                    extracted_stall = cv2.bitwise_and(frame, cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR))

                    # Find the bounding rectangle for the contour
                    x, y, w, h = cv2.boundingRect(contour)

                    # Crop the region from the original image using the bounding rectangle
                    parking_stall = extracted_stall[y:y + h, x:x + w]
                    parking_stall = cv2.cvtColor(parking_stall, cv2.COLOR_RGB2BGR)

                    predicted_label = helper_predict(vgg, model, parking_stall, input_size)
                    #resized = cv2.resize(cropped_mask, (SIZE, SIZE))
                    #resized = resized / 255
                    #input_data = np.expand_dims(resized, axis=0)
                    #predictions = Seq.predict(preprocess_input(input_data))
                    #print(predictions)
                    #predicted_labels = np.where(predictions > 0.5, 1, 0)
                    #print(predicted_labels)

                    if predicted_label == 0:
                        empty.append(contour)
                    else:
                        occupied.append(contour)

            # color the stall with according colors
            colored_image = cv2.drawContours(frame, empty, -1, GREEN, thickness=cv2.FILLED)
            colored_image = cv2.drawContours(colored_image, occupied, -1, RED, thickness=cv2.FILLED)

            alpha = 0.6
            result = cv2.addWeighted(overlay, alpha, colored_image, 1 - alpha, 0)

            # Put the text on the frame
            cv2.rectangle(result, (50, 100), (50 + 150, 100 - 20), (255, 255, 255), thickness=cv2.FILLED)
            cv2.putText(result, 'Available: ' + str(len(empty)), (50 + 10, 100 - 7), font, font_scale, (0, 0, 0), thickness)
            cv2.rectangle(result, (50, 130), (50 + 150, 130 - 20), (255, 255, 255), thickness=cv2.FILLED)
            cv2.putText(result, 'Occupied: ' + str(len(occupied)), (50 + 10, 130 - 7), font, font_scale, (0, 0, 0), thickness)

            cv2.imwrite(output_folder_path + '{}.jpg'.format(frame_count), result)
        frame_count += 1

        # Display or save the cropped image
        #cv2.imshow('Cropped Image', resized)
        #cv2.waitKey(0)

        # early stopping
        if frame_count == 500:
            break

    cap.release()
    cv2.destroyAllWindows()
    return

def convert_frames_to_video(frame_folder, output_path, frame_per_second):  
    '''
    combine the frames into a video
    input:
    frame_folder: str, path to the folder storing the frames
    output_path: str, path to where the video will be saved
    frame_per_second: int, the number of frames in a second

    return None
    '''
    file_names = os.listdir(frame_folder)
    # Sort the file names numerically, excluding non-numeric file names
    sorted_file_names = sorted(
        [f for f in file_names if f.split(".")[0].isdigit()],
        key=lambda x: int(x.split(".")[0])
    )

    # read the first frame to obtain its dimension
    frame_1_path = os.path.join(frame_folder, sorted_file_names[0])
    frame_1 = cv2.imread(frame_1_path)
    frame_width, frame_height, _ = frame_1.shape

    fps = frame_per_second  # Specify the frames per second
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the video codec (e.g., "mp4v", "XVID", etc.)
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for frame_file in sorted_file_names:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)

        # Resize the frame if necessary
        # frame = cv2.resize(frame, (frame_width, frame_height))
        success = output_video.write(frame)

        # Check if the frame was written successfully
        if not success:
            print(f"Error writing frame {frame_file}")

    output_video.release()
    cv2.destroyAllWindows()
    print('Video conversion complete.')
    return

def convert_frames_to_gif(frames_folder, output_path, duration=25):
    '''
    combining frames into gif
    input:
    frame_folder: str, path to the folder storing the frames
    output_path: str, path to where the gif will be saved
    frame_per_second: int, the duration of the gif

    return None
    '''
    # Get a list of all image files in the frames folder
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
                          key=lambda x: int(x.split(".")[0]))

    images = []
    for frame_file in frame_files[:200]:
        # Open each image frame
        frame_path = os.path.join(frames_folder, frame_file)
        image = Image.open(frame_path)

        # Add image to the list of frames
        images.append(image)

    # Save the frames as a GIF
    images[0].save(output_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

    print('GIF conversion complete.')
    return

