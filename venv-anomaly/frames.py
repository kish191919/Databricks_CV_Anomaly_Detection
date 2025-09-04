import cv2
from pathlib import Path

# Function to extract frames from a video file
def extract_frames(video_path:Path, output_dir:Path, duration:int=5) -> None:

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"{fps = }")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0

    success, image = cap.read()
    while success:
        if frame_count % fps == 0:
            frame_time = frame_count // fps
            output_path = output_dir / f"frame_{frame_time:04d}.jpg"
            cv2.imwrite(output_path, image)
            print(f"Saved {output_path}")
            # if frame_time >= duration:
            #     break
        
        success, image = cap.read()
        frame_count += 1


if __name__ == "__main__":
    video_path = Path('video.mp4')
    output_dir = Path('frames')
    extract_frames(video_path, output_dir)
