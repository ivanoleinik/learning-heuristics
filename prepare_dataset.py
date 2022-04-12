import os
import cv2
import math
import tqdm
import torch
import numpy as np


DATA = 'data/train maps'
TRAIN = 'data/train data/inputs'
TARGET4 = 'data/train data/target4'
TARGET8 = 'data/train data/target8'

SIZE = 64
SEED = 1337


def map2numpy(file):
    fin = open(os.path.join(DATA, file))
    lines = fin.readlines()
    info, data = lines[:4], lines[4:]
    h, w = map(lambda s: int(s.split()[1]), info[1:3])
    arr = np.zeros((h, w))
    for i, line in enumerate(data):
        for j in range(w):
            if line[j] != '.':
                arr[i][j] = 1
    return arr


def get_map_tiles(tile_size):
    tiles = []
    for path, _, files in os.walk(DATA):
        for file in files:
            arr = cv2.resize(map2numpy(file).astype(int), (SIZE, SIZE),
                             interpolation=cv2.INTER_NEAREST)
            for i in range(0, SIZE, tile_size):
                for j in range(0, SIZE, tile_size):
                    tile = arr[i:i + tile_size, j:j + tile_size]
                    tile_90 = np.rot90(tile)
                    tile_180 = np.rot90(tile_90)
                    tile_270 = np.rot90(tile_180)
                    tile_flip_ud = np.flipud(tile)
                    for t in [tile, tile.T, np.fliplr(tile), tile_flip_ud,
                              tile_90, tile_180, tile_270, np.rot90(tile_flip_ud)]:
                        tiles.append(t)
    return tiles


def is_valid_map(arr, x_goal, y_goal):
    area = SIZE * SIZE
    return 0.1 * area < np.count_nonzero(np.array(bfs(arr, x_goal, y_goal)) == -1) < 0.9 * area


def get_input_tensor(arr, x_goal, y_goal):
    pose_tensor = np.zeros((SIZE, SIZE, 1))
    pose_tensor[x_goal][y_goal] = [1]
    return torch.from_numpy(np.dstack((arr, pose_tensor)).transpose((2, 0, 1)))


def bfs(arr, x_goal, y_goal):
    dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
    n, m = len(arr), len(arr[0])
    dist = [[math.inf] * m for _ in range(n)]
    dist[x_goal][y_goal] = 0
    q, cur = [(x_goal, y_goal)], 0
    while cur < len(q):
        [x, y] = q[cur]
        cur += 1
        for k in range(len(dx)):
            xx, yy = x + dx[k], y + dy[k]
            if 0 <= xx < n and 0 <= yy < m and not arr[xx][yy] and dist[xx][yy] > dist[x][y] + 1:
                dist[xx][yy] = dist[x][y] + 1
                q.append((xx, yy))
    for i in range(n):
        for j in range(m):
            if dist[i][j] == math.inf:
                dist[i][j] = -1
    return dist


def dijkstra(arr, x_goal, y_goal):
    dx, dy = [1, 0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, 1, -1, 1, -1]
    n, m = len(arr), len(arr[0])
    dist = [[math.inf] * m for _ in range(n)]
    dist[x_goal][y_goal] = 0
    q = dict(((x, y), dist[x][y]) for y in range(SIZE) for x in range(SIZE))
    for _ in range(SIZE * SIZE):
        v = min(q, key=q.get)
        d = q[v]
        if d == math.inf:
            break
        del q[v]
        x, y = v
        for k in range(len(dx)):
            xx, yy = x + dx[k], y + dy[k]
            d = 1 if k < 4 else math.sqrt(2)
            if 0 <= xx < n and 0 <= yy < m and not arr[xx][yy] and dist[xx][yy] > dist[x][y] + d:
                dist[xx][yy] = dist[x][y] + d
                q[(xx, yy)] = dist[xx][yy]
    for i in range(n):
        for j in range(m):
            if dist[i][j] == math.inf:
                dist[i][j] = -1
    return dist


def gen_data(dataset_size=50000, p=2/3, max_iter=SIZE // 4):
    small_sz, big_sz = SIZE // 4, SIZE // 2
    small_tiles = get_map_tiles(small_sz)
    big_tiles = get_map_tiles(big_sz)
    n_small, n_big = len(small_tiles), len(big_tiles)
    cnt = 0
    pbar = tqdm.tqdm(total=dataset_size)
    while cnt < dataset_size:
        new_map = np.zeros((SIZE, SIZE))
        for i in range(0, SIZE, big_sz):
            for j in range(0, SIZE, big_sz):
                if np.random.rand() > p:
                    for ii in range(i, i + big_sz, small_sz):
                        for jj in range(j, j + big_sz, small_sz):
                            new_map[ii:ii + small_sz, jj:jj + small_sz] = small_tiles[np.random.randint(n_small)]
                else:
                    new_map[i:i + big_sz, j:j + big_sz] = big_tiles[np.random.randint(n_big)]
        ok = False
        for _ in range(max_iter):
            x_goal, y_goal = np.random.randint(0, SIZE, 2)
            if new_map[x_goal][y_goal] == 0 and is_valid_map(new_map, x_goal, y_goal):
                ok = True
                break
        if ok:
            torch.save(get_input_tensor(new_map, x_goal, y_goal), os.path.join(TRAIN, str(cnt)))
            torch.save(torch.tensor(bfs(new_map, x_goal, y_goal)), os.path.join(TARGET4, str(cnt)))
            torch.save(torch.tensor(dijkstra(new_map, x_goal, y_goal)), os.path.join(TARGET8, str(cnt)))
            cnt += 1
            pbar.update(1)


if __name__ == '__main__':
    np.random.seed(SEED)
    gen_data(dataset_size=50000)
