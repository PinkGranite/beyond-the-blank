import os
import time
import cv2
import numpy as np
from PIL import Image
from audio2text import Whisper
from dalle2 import DalleClient
import sounddevice as sd
import soundfile as sf
import threading
from pydub import AudioSegment
import assemblyai as aai
import time
import random

aai.settings.api_key = "b87ebe668b104115aea6d2f5ce11fd95"

# Constants
IMAGE_FOLDER = "static/images"
RATE = 44100
CHANNELS = 1
FORMAT = 'int16'
WAVE_OUTPUT_FILENAME = "output.wav"
MP3_OUTPUT_FILENAME = "output.mp3"

# Initialize DalleClient
dalle_client = DalleClient(api_key='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')

# Global variable to control recording
is_recording = False
recording_data = []

# Start recording in a separate thread
def record_audio_thread():
    global is_recording, recording_data
    recording_data = []

    def callback(indata, frames, time, status):
        if is_recording:
            recording_data.extend(indata.copy())
        else:
            raise sd.CallbackStop()

    print("Recording started...")
    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=FORMAT, callback=callback):
            while is_recording:
                sd.sleep(100)
    except sd.CallbackStop:
        pass

    # Save as WAV format first
    print("Saving as WAV file...")
    sf.write(WAVE_OUTPUT_FILENAME, np.array(recording_data), RATE)

    # Convert WAV to MP3
    print("Converting WAV to MP3...")
    audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
    audio.export(MP3_OUTPUT_FILENAME, format="mp3")
    print(f"Recording saved as MP3: {MP3_OUTPUT_FILENAME}")

# Start recording
def start_recording():
    global is_recording
    is_recording = True
    thread = threading.Thread(target=record_audio_thread)
    thread.start()

# Stop recording
def stop_recording():
    global is_recording
    is_recording = False

# Transcribe audio to text
def transcribe_audio():
    print("Transcribing audio...")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe('./output.mp3')

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
    else:
        print(transcript.text)
    transcription = transcript.text
    print(f"Transcription completed: {transcription}")
    return transcription

def pixelate_transition(current_array: np.ndarray, next_array: np.ndarray, screen, steps=30, delay=50):
    """
    Perform a pixelate transition between two images represented as NumPy arrays.

    Args:
        current_array (np.ndarray): The current image as a NumPy array.
        next_array (np.ndarray): The next image as a NumPy array.
        screen: The OpenCV window to display the transition.
        steps (int): Number of steps for the transition.
        delay (int): Delay in milliseconds between steps.
    """
    # Ensure input arrays are in the correct shape
    if len(current_array.shape) != 3 or len(next_array.shape) != 3:
        raise ValueError("Input images must have 3 dimensions (H, W, C).")

    # Get dimensions of the largest image
    max_height = max(current_array.shape[0], next_array.shape[0])
    max_width = max(current_array.shape[1], next_array.shape[1])

    # Resize both images to the same dimensions
    current_array_resized = cv2.resize(current_array, (max_width, max_height), interpolation=cv2.INTER_LINEAR)
    next_array_resized = cv2.resize(next_array, (max_width, max_height), interpolation=cv2.INTER_LINEAR)

    # Final safety check: Ensure both arrays have the same shape
    if current_array_resized.shape != next_array_resized.shape:
        raise ValueError(f"Shape mismatch after resizing: {current_array_resized.shape} vs {next_array_resized.shape}")

    for step in range(steps):
        # Calculate pixelation block size
        block_size = max(1, int((step + 1) * max(max_width, max_height) / (steps * 20)))

        # Pixelate current image
        current_resized = cv2.resize(current_array_resized, (max_width // block_size, max_height // block_size), interpolation=cv2.INTER_LINEAR)
        current_pixelated = cv2.resize(current_resized, (max_width, max_height), interpolation=cv2.INTER_NEAREST)

        # Pixelate next image
        next_resized = cv2.resize(next_array_resized, (max_width // block_size, max_height // block_size), interpolation=cv2.INTER_LINEAR)
        next_pixelated = cv2.resize(next_resized, (max_width, max_height), interpolation=cv2.INTER_NEAREST)

        # Blend the two images
        alpha = step / steps
        blended_image = cv2.addWeighted(current_pixelated, 1 - alpha, next_pixelated, alpha, 0)

        # Display the transition
        cv2.imshow(screen, blended_image)
        cv2.waitKey(delay)

    # Display the final image
    cv2.imshow(screen, next_array_resized)
    cv2.waitKey(delay)

# Pixelate transition
def pixelate_and_display(current_image, next_image, screen):
    current = cv2.imread(current_image)
    centered_current = center_image_on_canvas(current)
    next_img = cv2.imread(next_image)
    centered_next = center_image_on_canvas(next_img)
    pixelate_transition(centered_current, centered_next, screen)

def center_image_on_canvas(image, canvas_width=1080, canvas_height=1920):
    """
    Center an image on a black canvas with specified width and height.
    If the image width is smaller than the canvas width, it will be resized to fit the canvas width.
    Only adds black padding to the top and bottom.

    Args:
        image (numpy.ndarray): The image to be centered.
        canvas_width (int): Width of the canvas.
        canvas_height (int): Height of the canvas.

    Returns:
        numpy.ndarray: The resulting image centered on the canvas.
    """
    # Get the dimensions of the image
    img_height, img_width = image.shape[:2]

    # Resize image to fit the canvas width if needed
    if img_width < canvas_width:
        scaling_factor = canvas_width / img_width
        new_width = canvas_width
        new_height = int(img_height * scaling_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Get updated dimensions of the resized image
    img_height, img_width = image.shape[:2]

    # Create a blank canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Calculate the top-left corner position to center the image vertically
    y_offset = (canvas_height - img_height) // 2

    # Ensure no negative values for offsets (in case image height is larger than canvas)
    y_offset = max(0, y_offset)

    # Place the image on the canvas (vertically centered)
    canvas[y_offset:y_offset + img_height, :] = image[:min(img_height, canvas_height - y_offset), :]

    return canvas

# Update the main loop
def main():
    global is_recording
    images = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f)) and not f.endswith('_.png') and not f.startswith('generated_') and not f.startswith('edited_')]
    print(images)
    if not images:
        print("No images found in the directory.")
        return

    current_index = 0
    cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image Viewer", 1080, 1920)

    img_path = images[current_index]
    img = cv2.imread(img_path)
    centered_img = center_image_on_canvas(img)  # Center the image
    cv2.imshow("Image Viewer", centered_img)

    # Track last interaction time and image generation time
    last_interaction_time = time.time()
    last_generation_time = None
    
    while True:
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('1'):  # Record for 5 seconds
            if not is_recording:
                # Update centered_img to current image
                img = cv2.imread(images[current_index])
                centered_img = center_image_on_canvas(img)
                
                start_recording()
                start_time = time.time()
                for i in range(100):  # 100 steps progress bar
                    # Calculate actual time passed
                    elapsed = time.time() - start_time
                    if elapsed >= 5:
                        break
                        
                    # Display countdown and progress bar
                    countdown = int(5 - elapsed)
                    progress = int((elapsed/5) * 100)
                    temp_img = centered_img.copy()
                    
                    # Draw progress bar
                    bar_width = 400
                    bar_height = 30
                    filled_width = int(bar_width * progress/100)
                    cv2.rectangle(temp_img, (50, 80), (50+bar_width, 80+bar_height), (255,255,255), 2)
                    cv2.rectangle(temp_img, (50, 80), (50+filled_width, 80+bar_height), (0,255,0), -1)
                    
                    # Draw text with thicker font
                    cv2.putText(temp_img, f"Recording... {countdown}s remaining", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Image Viewer", temp_img)
                    cv2.waitKey(1)
                    time.sleep(0.05)  # Smooth animation
                
                stop_recording()
                time.sleep(1)
                
                # Show transcribing message
                temp_img = centered_img.copy()
                cv2.putText(temp_img, "Transcribing audio...", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Image Viewer", temp_img)
                cv2.waitKey(1)
                
                prompt = transcribe_audio()
                print(f"Transcription: {prompt}")
                cv2.waitKey(1)
                
                if prompt != "":
                    # Show generating message
                    temp_img = centered_img.copy()
                    cv2.putText(temp_img, f"Transcribed: {prompt}", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(temp_img, "Generating new painting, please wait", 
                              (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Image Viewer", temp_img)
                    cv2.waitKey(1)
                    
                    dalle_client.inpaint(images[current_index], prompt)
                    pixelate_and_display(images[current_index], './generated_image.png', "Image Viewer")
                    last_generation_time = time.time()
            last_interaction_time = time.time()
            
        # Auto switch image after 30s of no interaction
        if time.time() - last_interaction_time > 30 and not last_generation_time:
            random_index = random.randint(0, len(images) - 1)
            pixelate_and_display(images[current_index], images[random_index], "Image Viewer")
            current_index = random_index
            last_interaction_time = time.time()
            
        # Auto switch image 15s after generation if no interaction
        if last_generation_time and time.time() - last_generation_time > 15 and time.time() - last_interaction_time > 15:
            random_index = random.randint(0, len(images) - 1)
            pixelate_and_display(images[current_index], images[random_index], "Image Viewer")
            current_index = random_index
            last_interaction_time = time.time()
            last_generation_time = None

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()