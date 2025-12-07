import os
from video_utils import read_video
from tracker import Tracker


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'Data', 'ANPR-ATCC')
    input_video_path = os.path.join(data_dir, 'anpr_atcc.mp4')

    frames = read_video(input_video_path)
    track = Tracker()
    _ = track.process_video(frames)


if __name__ == '__main__':
    main()

