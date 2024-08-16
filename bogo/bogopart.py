import os
import contextlib
import time
from collections import defaultdict
import numpy as np
from humanize.filesize import naturalsize

if os.uname().nodename == "decid":
    # oomkill us before killing chrome tabs
    open("/proc/self/oom_score_adj", "w").write("1000")

info = open("/home/sylv/r8/cog-sdxl/meta.csv", "r").read()
ass, *tensors = info.strip().split("\n")
sizes = [int(t.split(" ")[-1]) for t in tensors if t]
ffd_bins = [
    [sizes[i] for i in map(int, thread.split(";"))] for thread in ass.split(",")
]


def natsize(val):
    return naturalsize(val, True, format="%.3f")


SZ_COEF = 0.03 / 1024
LN_COEF = 450
INTERCEPT = -3100


def emperical(size: int, n_files: int) -> float:
    return size * SZ_COEF + n_files * LN_COEF #+ INTERCEPT


def estimate_tuple(tup: tuple[int, int]) -> float:
    return tup[0] * SZ_COEF + tup[1] * LN_COEF


def estimate(bin: list[int]) -> float:
    return emperical(sum(bin), len(bin))


@contextlib.contextmanager
def timer(msg: str) -> "Iterator[None]":
    start = time.time()
    yield
    print(f"{msg} took {time.time() - start:.3f}s")


# @timer("multidim bin sum")
# def vectorized_multi_dim_sum(sizes, bin_matrix):
#     sizes = np.array(sizes)
#     bin_matrix = np.array(bin_matrix)

#     # Number of bins
#     num_bins = np.max(bin_matrix) + 1

#     # Reshape sizes for broadcasting: (1, len(sizes), 1)
#     expanded_sizes = sizes.reshape((1, -1, 1))

#     # Create an array of bin indices: (num_bins, 1, 1)
#     bin_indices = np.arange(num_bins).reshape((-1, 1, 1))

#     # Broadcast bin_matrix against bin_indices: (1048576, 131, num_bins)
#     # and then check equality, resulting in a boolean mask
#     masks = bin_matrix[..., np.newaxis] == bin_indices

#     # Perform element-wise multiplication and sum over the second axis (sizes)
#     sums = (expanded_sizes * masks).sum(axis=1)

#     return sums

n_bins = 20


def unbin(binning):
    bins = [[] for i in range(n_bins)]
    for idx, bin in enumerate(binning):
        bins[bin].append(sizes[idx])
    return bins


# def improve(bins: list[list[int]]):
#     bin_sizes = list(map(sum, bins))
#     bin_lengths = list(map(len, bins))

#     def step(key):
#         # take the worst bin
#         # move an item into another bin, such that
#         # total estimate makespan is reduced
#         # max(map(estimate, bin_size_and_length)) < old
#         # where bin_sizes_and_length is a copy with the items moved
#         # optimize it later
#         nonlocal bin_sizes, bin_lengths
#         worst_bin = max(bins, key=estimate)
#         worst_idx = bins.index(worst_bin)
#         makespan = estimate(worst_bin)
#         bin_sizes_and_length = [[sum(bin), len(bin)] for bin in bins]
#         makespan = max(map(estimate_tuple, bin_sizes_and_length))
#         # assume we'll remove smth
#         bin_lengths[worst_idx] -= 1
#         for item in sorted(worst_bin, reverse=True):
#             bin_sizes[worst_idx] -= item
#             for other_bin in sorted(bins, key=estimate, reverse=True):
#                 if other_bin == worst_bin:
#                     continue

#                 # candidate = other_bin + [item]
#                 # max(removed_makespan, ...)
#                 new_size = sum(other_bin) + item
#                 new_len = len(other_bin) + 1


#             desc = sorted(bins, key=estimate, reverse=True)
#             for i, bin in enumerate(desc):
#                 new_size = sum(bin) + item
#                 new_len = len(bin) + 1
#                 allowed = new_size <= max(bin_sizes) and new_len <= max(bin_lengths)
#                 better = new_size < max(bin_sizes) or new_len < max(bin_lengths)
#                 if allowed and better:
#                     big_bin.remove(item)
#                     bin.append(item)
#                     bin_sizes = list(map(sum, bins))
#                     bin_lengths = list(map(len, bins))
#                     return True
#         return False

#     steps = 0
#     while 1:
#         imp1 = step(sum)
#         imp2 = step(len)
#         steps += imp1 + imp2
#         if not imp1 and not imp2:
#             break
#     print("improved with", steps, "steps")


def improve(bins: list[list[int]]):
    initial_makespan = max(map(estimate, bins))

    def step():
        worst_bin = max(bins, key=estimate)
        for item in sorted(worst_bin, reverse=True):
            worst_bin.remove(item)
            for other_bin in sorted(bins, key=estimate):
                if other_bin == worst_bin:
                    continue
                other_bin.append(item)
                new_makespan = max(map(estimate, bins))
                if new_makespan < initial_makespan:
                    return True
                other_bin.remove(item)
            worst_bin.append(item)
        return False

    for i in range(1000000):
        if not step():
            break
    improvement = max(map(estimate, bins)) - initial_makespan
    print(f"improved with {i} steps. improved makespan by {improvement:.2f}")


def vectorized_bins_estimate(sizes, bin_matrix):
    sizes = np.array(sizes)
    bin_matrix = np.array(bin_matrix)

    # The number of bins is determined by the unique elements in the bin matrix
    num_bins = np.max(bin_matrix) + 1

    # Create a boolean mask for each bin
    masks = np.array([(bin_matrix == i) for i in range(num_bins)])

    # Use broadcasting to multiply sizes with masks and sum along the appropriate axis
    # Results in a 2D array where each row corresponds to a bin and each column to a sum in that bin
    with timer("sum"):
        sums = np.array([sizes * mask for mask in masks]).sum(axis=2).T
        # sums = (sizes[..., np.newaxis, np.newaxis] * masks.transpose(2, 0, 1)).sum(axis=-1).T
    
    # calculate number of items in each bin
    with timer("len"):
        lengths = masks.sum(axis=2).T
    
    # for each bin, add the two estimate components
    # we need to operate on the second dimension
    estimates = sums * SZ_COEF + lengths * LN_COEF
    return estimates


best_sizes = []
best = []
for i in range(10):
    sizes = [int(t.split(" ")[-1]) for t in tensors if t]
    with timer("make random"):
        bin_mat = np.random.randint(0, n_bins, [1024 * 256, len(sizes)], dtype=np.int8)

    estimates = vectorized_bins_estimate(sizes, bin_mat).max(axis=1)
    batch_best = estimates.argsort()[:4]
    best.extend(estimates[batch_best])
    best_sizes.extend(bin_mat[batch_best])
    print(natsize(estimates.min()))

print("done")
print(natsize(min(best)))

attempts = []
for binning in best_sizes:
    bins = unbin(binning)
    improve(bins)
    attempts.append(bins)

final_best = min(attempts, key=lambda bins: max(map(estimate, bins)))


def to_idx(sizes: list[int], bins: list[list[int]]) -> list[list[int]]:
    indx = defaultdict(list)
    for i, size in enumerate(sizes):
        indx[size].append(i)
    result = [[indx[size].pop(0) for size in bin] for bin in bins]
    for bin in indx.values():
        assert len(bin) == 0, f"unexpected remaining tensor {len(bin)}"
    return result


for bin in final_best:
    print("size", natsize(sum(bin)), "len", len(bin))
initial_makespan = best[attempts.index(final_best)]
print("estimated makespan", natsize(max(map(estimate, final_best))))
print(",".join([";".join(map(str, bin)) for bin in to_idx(sizes, final_best)]))
