from collections import Counter, defaultdict
import random
import functools

# search partitioning space in advance to find optimal thread packing
# we want tensors of the same size to be together while making sure each thread has similar amounts of work
# we can do this by sorting the tensors by size and then assigning them to threads in order
# when we can we also want to assign work to the thread with the least amount of work

# find the bin packing with the shortest max bin size
# then, find the bin packing with the highest number of repeated sizes with the same max bin size

# if the the entire slowest thread is one tensor and cannot be packed differently,
# all the other bins can use that to pack tensors of the same size


def ffd_binpack(sizes: list[int], capacity: int) -> list[list[int]]:
    bins: list[list[int]] = []
    bin_sizes: list[int] = []
    for number in sizes:
        possible_inserts = (
            i for i, size in enumerate(bin_sizes) if size + number <= capacity
        )
        if insert := next(possible_inserts, None):
            bins[insert].append(number)
            bin_sizes[insert] += number
        else:
            bins.append([number])
            bin_sizes.append(number)
    return bins


def multiway_partition(sizes: list[int], bins: int = 32, k=20) -> list[list[int]]:
    lower_bound_capcity = int(max(sum(sizes) / len(sizes), max(sizes)))
    upper_bound_capcity = max(2 * sum(sizes) / len(sizes), max(sizes))
    for i in range(k):
        capacity = round((upper_bound_capcity + lower_bound_capcity) / 2)
        solution = ffd_binpack(sizes, capacity)
        if len(solution) <= bins:
            upper_bound_capcity = capacity
        else:
            lower_bound_capcity = capacity
        if upper_bound_capcity - lower_bound_capcity < 10:
            break
    solution = ffd_binpack(sizes, upper_bound_capcity)
    # assert len(solution) == 32
    return solution


def partition(sizes: list[int], bins: int = 32) -> list[list[int]]:
    indexes = [[] for _ in range(bins)]
    bin_sizes = [0] * bins

    lemgthy = max(sizes)

    initial = sorted(enumerate(sizes), key=lambda x: x[1], reverse=True)
    for i, size in initial:
        smallest = bin_sizes.index(min(bin_sizes))
        bin_sizes[smallest] += size
        indexes[smallest].append(i)

    while 1:
        smallest, biggest = map(bin_sizes.index, (min(bin_sizes), max(bin_sizes)))
        diff = bin_sizes[biggest] - bin_sizes[smallest]
        _, index_to_move = max(
            [(sizes[i], i) for i in indexes[biggest] if sizes[i] * 2 < diff],
            default=(None, None),
        )
        if not index_to_move:
            break
        indexes[biggest].remove(index_to_move)
        bin_sizes[biggest] -= sizes[index_to_move]
        indexes[smallest].append(index_to_move)
        bin_sizes[smallest] += sizes[index_to_move]

    def indexes_to_sizes(indx: list[list[int]]) -> list[list[int]]:
        return [[sizes[i] for i in bin] for bin in indx]

    counts = [Counter([sizes[i] for i in bin]) for bin in indexes]
    size_bins = indexes_to_sizes(indexes)

    for bin_number, bin in enumerate(bin_number, size_bins):
        for i, item in enumerate(bin):
            for replacement_bin in size_bins:
                if replacement_bin[item] > bin[item]:
                    pass  # swap

    return indexes


def score(binning: list[list[int]]) -> tuple[int, int]:
    longest = max(map(sum, binning))
    changes = 0
    for bin in binning:
        for prev_size, next_size in zip(bin[:-1], bin[1:]):
            if prev_size != next_size:
                changes += 1
    changes = sum(
        1
        for prev_size, next_size in zip(bin[:-1], bin[1:])
        for bin in binning
        if prev_size != next_size
    )
    return (longest, changes)


# minimize the sum of the largest size in each bin
# that is, few bins with the biggest sizes
# instead of changes 

def greedy_partition(sizes: tuple[int], n_bins: int = 32) -> list[list[int]]:
    sizes = list(sizes)
    random.shuffle(sizes)
    bins = [[] for _ in range(n_bins)]
    bin_sizes = [0] * n_bins

    for number in sizes:
        current_makespan = max(bin_sizes)
        candidates = []
        for i, bin in enumerate(bins):
            # of threads that are bigger max size than this, take the least full one
            # otherwise take the one this increases the least
            new_size = bin_sizes[i] + number
            makespan_increase = max(0, new_size - current_makespan)
            candidates.append((makespan_increase, max(number - max(bin, default=0), 0), bin_sizes[i], i))
        i = min(candidates)[-1]
        bins[i].append(number)
        bin_sizes[i] += number

    # move the smallest tensor from the largest thread with multiple tensors into any empty threads
    while any(not b for b in bins):
        biggest = sorted([bin for bin in bins if len(bin) > 2], key=sum)[-1]
        empty = next(i for i, bin in enumerate(bins) if not bin)
        moved = min(biggest)
        biggest.remove(moved)
        bins[empty].append(moved)
        # for bin in bins:
        #     for item in bin:
        #         if item < moved and sum(bin) + item < max_len:
        #             bin.remove(item)
        #             empty.append

    return bins

@functools.cache
def massage(sizes: tuple[int], n_bins: int = 32) -> list[list[int]]:
    indx = defaultdict(list)
    for i, size in enumerate(sizes):
        indx[size].append(i)

    ordered = sorted(sizes, reverse=True)
    #bins = greedy_partition(sizes, n_bins)
    bins = multiway_partition(sizes, n_bins)
    assert sum(map(len, bins)) == len(sizes), "wrong"

    result = [[indx[size].pop(0) for size in bin] for bin in bins]
    for bin in indx.values():
        assert len(bin) == 0, "unexpected remaining tensor " + len(bin)
    return result

def greedy_partition_rand(
    sizes: list[int], n_bins: int = 32, k: int = 4
) -> list[list[int]]:
    bins = [[] for _ in range(n_bins)]
    bin_sizes = [0] * n_bins

    for number in sizes:
        current_makespan = max(bin_sizes)
        candidates = []
        for i, bin in enumerate(bins):
            new_size = bin_sizes[i] + number
            makespan_increase = max(0, new_size - current_makespan)
            candidates.append((makespan_increase, number in bin, i))
        _, _, i = random.choice(sorted(candidates)[:k])
        bins[i].append(number)
        bin_sizes[i] += number
    return bins


def fuzzy(sizes, n_bins: int = 32, tries: int = 10) -> list[list[int]]:
    best = greedy_partition_rand(sizes)
    best_score = score(best)
    for i in range(tries):
        solution = greedy_partition_rand(sizes)
        new_score = score(solution)
        if new_score > best_score:
            best_score = new_score
            best = solution
    return best
