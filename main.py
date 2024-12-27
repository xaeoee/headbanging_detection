import argparse
from headbanging import *

def main():
    """
    Main function to parse arguments and initiate headbanging detection.
    """
    parser = argparse.ArgumentParser(description="Headbanging Detection using Mediapipe and OpenCV.")

    parser.add_argument("--angle_threshold", type=float, default=4.3, help="Threshold angle difference to detect headbanging.")
    parser.add_argument("--time_threshold", type=float, default=0.6, help="Time interval threshold for direction changes.")
    parser.add_argument("--headbanging_threshold", type=int, default=4, help="Number of direction changes to confirm headbanging.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    parser.add_argument("--mode", type=str, choices=["webcam", "video"], default="webcam", help="Mode of operation: 'webcam' for live detection, 'video' for processing a video file.")
    parser.add_argument("--video_path", type=str, default=None, help="Path to the video file (required if mode is 'video').")

    args = parser.parse_args()

    if args.mode == "video" and not args.video_path:
        parser.error("--video_path is required when mode is set to 'video'.")

    # Handle mode selection
    if args.mode == "webcam":
        print("Running in webcam mode.")
        video_path = 0

    elif args.mode == "video":
        print(f"Running in video mode with file: {args.video_path}")
        video_path=args.video_path
    
    print(args.angle_threshold, args.time_threshold, args.headbanging_threshold, args.verbose, video_path)

    headbanging_detection(
        angle_threshold=args.angle_threshold,
        time_threshold=args.time_threshold,
        headbanging_threshold=args.headbanging_threshold,
        verbose=args.verbose,
        video_path=video_path
    )

if __name__ == "__main__":
    main()
