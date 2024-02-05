"""
Test the saved model on the ship data and print the evaluation metrics
"""
import motmetrics as mm
from Test import *
from tqdm.auto import tqdm


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
        adj_mat, current_id = test_generator(seq_id, frame_id)

        # Call update once for per frame.
        acc.update(
                current_id,  # Ground truth objects in this frame
                current_id,  # Detector hypotheses in this frame
                adj_mat.cpu().numpy()
                )


mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'idf1'], name='acc')
strsummary = mm.io.render_summary(
    summary,
    formatters={'mota' : '{:.2%}'.format,
                'idf1' : '{:.2%}'.format,},
    namemap={'mota': 'MOTA', 'idf1' : 'IDF1'}
)
print(strsummary)
