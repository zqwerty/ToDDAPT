import numpy as np

trials = 10000
turns = 30
starts = np.random.randint(0, turns, size=(trials))
ends = np.random.randint(0, turns, size=(trials))
turn_dist = np.zeros((turns))
for start, end in zip(starts, ends):
    # for i in range(min(start, end), max(start, end)+1):
    #     turn_dist[i] += 1
    for i in range(0, end+1):
        turn_dist[i] += 1
    for i in range(end, turns):
        turn_dist[i] += 1
    # r = np.random.randint(0, end + 1)
    # for i in range(r, end+1):
    #     turn_dist[i] += 1
    # for i in range(0, r+1):
    #     turn_dist[i] += 1
    # for i in range(np.random.randint(0, r + 1), r):
    #     turn_dist[i] += 1
print(turn_dist/turn_dist.sum())

