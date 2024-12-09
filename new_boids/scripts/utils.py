import os
from matplotlib.legend_handler import HandlerTuple

# Again from https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/signal/train_test_split.py
def temporal_signal_split(data_iterator, train_ratio=0.7):
    train_snapshots = int(data_iterator.snapshot_count * train_ratio)
    train_iterator = data_iterator[0:train_snapshots]
    test_iterator = data_iterator[train_snapshots:]
    return train_iterator, test_iterator

# Utility function to create directories
def create_dirs(directory_structure):
    for path in directory_structure:
        os.makedirs(path, exist_ok=True)

class HandlerTupleVertical(HandlerTuple):
    def __init__(self, **kwargs):
        HandlerTuple.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # How many lines are there.
        numlines = len(orig_handle)
        handler_map = legend.get_legend_handler_map()

        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        height_y = (height / numlines)

        leglines = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)

            legline = handler.create_artists(legend, handle,
                                             xdescent,
                                             (4*i + 1)*height_y,
                                             width,
                                             2.5*height,
                                             fontsize, trans)
            leglines.extend(legline)

        return leglines