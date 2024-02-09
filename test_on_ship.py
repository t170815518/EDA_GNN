"""
Test the saved model on the ship data and print the evaluation metrics
"""
import motmetrics as mm
import numpy as np
from motmetrics.lap import linear_sum_assignment
from Test import *
from tqdm.auto import tqdm


def interpret_adj_mat(adj_mat):
    """
    Interprets the association result from the adjacency matrix (adj_mat).
    :param adj_mat: The adjacency matrix representing distances or dissimilarities.
    :return: gt_objects, hypothesis_objects, distance_mat
    """
    # Initialize lists for ground truth objects (gt_objects) and hypothesis objects (hypothesis_objects)
    gt_objects = []
    hypothesis_objects = []

    # Mask to keep track of available rows and columns
    available_rows = np.ones(adj_mat.shape[0], dtype=bool)
    available_cols = np.ones(adj_mat.shape[1], dtype=bool)

    # Store a copy of the original adj_mat for distance retrieval
    distance_mat = np.ones_like(adj_mat) * np.inf

    while True:
        # Find the maximum value in the matrix and its indices
        x_max = adj_mat.max()
        if x_max <= 0:
            break

        i, j = np.unravel_index(adj_mat.argmax(), adj_mat.shape)
        if i != j:
            print(f'[WARNING] i ({i}) != j ({j})')

        # Add the indices to ground truth and hypothesis lists and store the distance
        gt_objects.append(i)
        hypothesis_objects.append(j)
        distance_mat[i, j] = 1

        # Mark the chosen element in adj_mat as negative to avoid re-selection
        adj_mat[i, :] = -np.inf
        adj_mat[:, j] = -np.inf

    return list(range(adj_mat.shape[0])), list(range(adj_mat.shape[1])), distance_mat


TRAINED_MODEL_PATH = 'result/Feb-04-at-17-48-net_1024/net_1024.pth' # todo: replace it when using other models

# prepare the test Generator
test_generator = TestGenerator('', '1010new_processed.csv', is_ship=True,
                               net_path=TRAINED_MODEL_PATH)

# Create an accumulator that will be updated during each frame
acc = mm.MOTAccumulator(auto_id=True)

# iterate over each test sequence
for seq_id, seq in enumerate(test_generator.sequence):
    frame_len = seq.frame_len
    # iterate over each frame
    for frame_id in tqdm(range(5, frame_len)):
        adj_mat, current_id = test_generator(seq_id, frame_id)  # 如何理解adj_mat请参考论文3.5节Association Result Interpretation
        gt_objects, hypothesis_objects, distance_mat = interpret_adj_mat(adj_mat.cpu().numpy())
        acc.update(
                gt_objects,  # Ground truth objects in this frame
                hypothesis_objects,  # Detector hypotheses in this frame
                distance_mat
                )
        print(f'[frame_id={frame_id}]gt_objects={gt_objects}\thypothesis_objects={hypothesis_objects}')


mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'idf1'], name='acc')
strsummary = mm.io.render_summary(
    summary,
    formatters={'mota' : '{:.2%}'.format,
                'idf1' : '{:.2%}'.format,},
    namemap={'mota': 'MOTA', 'idf1' : 'IDF1'}
)
print(strsummary)
