# Let 𝑆 be a set of 𝑁 points from some metric space and, for each 𝑝 ∈ 𝑆,
# let 𝐵𝑆(𝑝,𝑟) denote the set of points of 𝑆 at distance at most 𝑟 from 𝑝.
# For given parameters 𝑀,𝐷 > 0 , an (𝑀,𝐷)-outlier (w.r.t. 𝑆) is a point 𝑝 ∈ 𝑆
# such that |𝐵𝑆(𝑝,𝐷)| ≤ 𝑀.

# Given 𝑆,𝑀, and 𝐷 , mark each point 𝑝 ∈ 𝑆 as outlier,
# if it is an (𝑀,𝐷)-outlier, and non-outlier otherwise.
import math

def euclidean(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def exact_outlier(S: list, D: float, M: int):
    distances = dict()
    outliers = list()
    for p in S:
        distances[p] = 0
        for r in S:
            if r != p:
                dist = euclidean(p, r)
                if dist <= D:
                    distances[p] += 1
        if distances[p] <= M:
            outliers.append(p)
    return outliers

def main():
    S = list()
    with open('test.txt', 'r') as f:
         lines = f.read().splitlines()
    for element in lines:
        x, y = element.split(',')
        point = (float(x), float(y))
        S.append(point)

    print(exact_outlier(S, 1.5, 3))
    print(exact_outlier(S, 2, 3))

if __name__ == '__main__':
    main()
