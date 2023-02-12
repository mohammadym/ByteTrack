import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from trackers.bytetrack.byte_tracker import BYTETracker


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('video')
    # Passed to ByteTrack
    parser.add_argument('--track_thresh', type=float, default=0.6)
    parser.add_argument('--match_thresh', type=float, default=0.9)
    parser.add_argument('--track_buffer', type=int, default=30)
    parser.add_argument('--frame_rate', type=int, default=30)
    parser.add_argument('--mot20', action='store_true')
    parser.add_argument('--true-size', type=int, nargs=2, default=(1080, 1920))
    parser.add_argument('--pred-size', type=int, nargs=2, default=(1080, 1920))
    args = parser.parse_args()

    output_dir = pathlib.Path('outputs')
    os.makedirs(output_dir, exist_ok=True)

    source_dir = pathlib.Path(args.data_dir) / args.video
    assert source_dir.exists(), "%s doesn't exist" % source_dir
    names = ('frame', 'track', 'left', 'top', 'width', 'height', 'score')
    df = pd.read_csv(source_dir / 'dets.txt', names=names)

    df['right'] = df.left + df.width
    df['bottom'] = df.top + df.height

    tracker = BYTETracker(
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        frame_rate=args.frame_rate,
        mot20=args.mot20,
    )

    start, end = df.frame.min(), df.frame.max()

    outputs = []

    for frame in tqdm.trange(start, end + 1):

        detections = df[df.frame == frame]
        # Seems like ByteTrack expects the detections to be in a TLBR format.
        # https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py#L189
        detections = detections[['top', 'left', 'bottom', 'right', 'score']]
        online_targets = tracker.update(detections.values, args.true_size, args.pred_size)

        for s_track in online_targets:

            top, left, width, height = s_track.tlwh
            line = (frame, s_track.track_id, left, top, width, height, s_track.score, 0, 0, 0)
            outputs.append(line)

    filename = output_dir / f'{args.video}.txt'
    np.savetxt(filename, np.asarray(outputs), fmt='%g', delimiter=',')
    filename = source_dir / 'pred' / 'pred.txt'
    np.savetxt(filename, np.asarray(outputs), fmt='%g', delimiter=',')
