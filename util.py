from PIL import Image
import os
import cv2

def convert_frames_to_video(frame_folder, output_path, frame_per_second):
    file_names = os.listdir(frame_folder)
    # Sort the file names numerically, excluding non-numeric file names
    sorted_file_names = sorted(
        [f for f in file_names if f.split(".")[0].isdigit()],
        key=lambda x: int(x.split(".")[0])
    )

    # read the first frame to obtain its dimension
    frame_1_path = os.path.join(frame_folder, sorted_file_names[0])
    frame_1 = cv2.imread(frame_1_path)
    frame_width, frame_height = frame_1.shape

    fps = frame_per_second  # Specify the frames per second
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the video codec (e.g., "mp4v", "XVID", etc.)
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for frame_file in sorted_file_names:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)

        # Resize the frame if necessary
        # frame = cv2.resize(frame, (frame_width, frame_height))
        output_video.write(frame)

    output_video.release()
    cv2.destroyAllWindows()
    print('Video conversion complete.')

def convert_frames_to_gif(frames_folder, output_path, duration=100):
    # Get a list of all image files in the frames folder
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    images = []
    for frame_file in frame_files:
        # Open each image frame
        frame_path = os.path.join(frames_folder, frame_file)
        image = Image.open(frame_path)

        # Add image to the list of frames
        images.append(image)

    # Save the frames as a GIF
    images[0].save(output_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

    print('GIF conversion complete.')

# Example usage
#frames_folder = 'frames'
#output_path = 'output.gif'
#duration = 100  # milliseconds per frame

# Convert the frames to GIF
#convert_frames_to_gif(frames_folder, output_path, duration)