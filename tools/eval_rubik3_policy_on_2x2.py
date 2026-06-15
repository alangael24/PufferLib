import argparse
from pathlib import Path

import numpy as np
import torch

from pufferlib.models import DefaultDecoder, DefaultEncoder, MinGRU, Policy


FACES = 6
FACE_STICKERS = 9
STICKERS = FACES * FACE_STICKERS
OBS_SIZE = STICKERS * FACES
ACTIONS = 18
U, R, F, D, L, B = range(6)

CORNER_POS = (0, 2, 6, 8)
CORNER_IDXS = tuple(face * FACE_STICKERS + pos for face in range(FACES) for pos in CORNER_POS)
SOLVED = np.array([face for face in range(FACES) for _ in range(FACE_STICKERS)], dtype=np.uint8)


def native_bin_to_torch_state(raw_path, policy):
    raw = np.fromfile(raw_path, dtype=np.float32)
    state = policy.state_dict()

    hidden = state["encoder.encoder.weight"].shape[0]
    obs = state["encoder.encoder.weight"].shape[1]
    actions = state["decoder.decoder.weight"].shape[0]
    layers = len([k for k in state if k.startswith("network.layers.")])

    expected = hidden * obs + (actions + 1) * hidden + layers * (3 * hidden * hidden)
    if raw.size != expected:
        raise RuntimeError(f"raw checkpoint has {raw.size} floats, expected {expected}")

    idx = 0
    new_state = {}

    n = hidden * obs
    new_state["encoder.encoder.weight"] = torch.from_numpy(
        raw[idx:idx + n].reshape(hidden, obs).copy())
    idx += n
    new_state["encoder.encoder.bias"] = torch.zeros_like(state["encoder.encoder.bias"])

    n = (actions + 1) * hidden
    fused_decoder = raw[idx:idx + n].reshape(actions + 1, hidden)
    idx += n
    new_state["decoder.decoder.weight"] = torch.from_numpy(fused_decoder[:actions].copy())
    new_state["decoder.decoder.bias"] = torch.zeros_like(state["decoder.decoder.bias"])
    new_state["decoder.value_function.weight"] = torch.from_numpy(
        fused_decoder[actions:actions + 1].copy())
    new_state["decoder.value_function.bias"] = torch.zeros_like(
        state["decoder.value_function.bias"])

    for layer in range(layers):
        key = f"network.layers.{layer}.weight"
        n = state[key].numel()
        new_state[key] = torch.from_numpy(raw[idx:idx + n].reshape(state[key].shape).copy())
        idx += n

    policy.load_state_dict(new_state)


def make_policy(checkpoint, hidden_size, num_layers):
    policy = Policy(
        DefaultEncoder(OBS_SIZE, hidden_size),
        DefaultDecoder([ACTIONS], hidden_size),
        MinGRU(hidden_size=hidden_size, num_layers=num_layers),
    )
    native_bin_to_torch_state(checkpoint, policy)
    policy.eval()
    return policy


def sticker_to_coord(idx):
    face = idx // FACE_STICKERS
    pos = idx % FACE_STICKERS
    row = pos // 3
    col = pos % 3

    if face == U:
        return col - 1, 1, row - 1, 0, 1, 0
    if face == R:
        return 1, 1 - row, 1 - col, 1, 0, 0
    if face == F:
        return col - 1, 1 - row, 1, 0, 0, 1
    if face == D:
        return col - 1, -1, 1 - row, 0, -1, 0
    if face == L:
        return -1, 1 - row, col - 1, -1, 0, 0
    return 1 - col, 1 - row, -1, 0, 0, -1


def coord_to_sticker(x, y, z, nx, ny, nz):
    if ny == 1:
        face, row, col = U, z + 1, x + 1
    elif ny == -1:
        face, row, col = D, 1 - z, x + 1
    elif nz == 1:
        face, row, col = F, 1 - y, x + 1
    elif nz == -1:
        face, row, col = B, 1 - y, 1 - x
    elif nx == 1:
        face, row, col = R, 1 - y, 1 - z
    else:
        face, row, col = L, 1 - y, z + 1
    return face * FACE_STICKERS + row * 3 + col


def rotate_vec(axis, turns, x, y, z):
    turns %= 4
    for _ in range(turns):
        ox, oy, oz = x, y, z
        if axis == 0:
            x, y, z = ox, -oz, oy
        elif axis == 1:
            x, y, z = oz, oy, -ox
        else:
            x, y, z = -oy, ox, oz
    return x, y, z


def action_to_layer(action):
    face = action // 3
    turn = action % 3
    if face == U:
        axis, layer = 1, 1
    elif face == R:
        axis, layer = 0, 1
    elif face == F:
        axis, layer = 2, 1
    elif face == D:
        axis, layer = 1, -1
    elif face == L:
        axis, layer = 0, -1
    else:
        axis, layer = 2, -1

    if turn == 0:
        turns = -layer
    elif turn == 1:
        turns = layer
    else:
        turns = 2
    return axis, layer, turns


def turn_2x2_projected(cube, action):
    axis, layer, turns = action_to_layer(int(action))
    out = cube.copy()
    for idx in CORNER_IDXS:
        x, y, z, nx, ny, nz = sticker_to_coord(idx)
        coord = x if axis == 0 else y if axis == 1 else z
        if coord != layer:
            continue
        x, y, z = rotate_vec(axis, turns, x, y, z)
        nx, ny, nz = rotate_vec(axis, turns, nx, ny, nz)
        dst = coord_to_sticker(x, y, z, nx, ny, nz)
        out[dst] = cube[idx]
    return out


def scramble_2x2_projected(depth, rng):
    cube = SOLVED.copy()
    last_face = -1
    actions = []
    for _ in range(depth):
        face = int(rng.integers(0, FACES))
        while depth > 1 and face == last_face:
            face = int(rng.integers(0, FACES))
        action = face * 3 + int(rng.integers(0, 3))
        cube = turn_2x2_projected(cube, action)
        actions.append(action)
        last_face = face
    return cube, actions


def is_solved(cube):
    return bool(np.all(cube[list(CORNER_IDXS)] == SOLVED[list(CORNER_IDXS)]))


def corner_distance(cube):
    return int(np.sum(cube[list(CORNER_IDXS)] != SOLVED[list(CORNER_IDXS)]))


def encode_obs(cubes):
    obs = np.zeros((len(cubes), OBS_SIZE), dtype=np.uint8)
    for row, cube in enumerate(cubes):
        obs[row, np.arange(STICKERS) * FACES + cube] = 1
    return torch.from_numpy(obs)


def evaluate_depth(policy, depth, episodes, max_steps, rng, sampled):
    cubes = []
    scramble_actions = []
    for _ in range(episodes):
        cube, actions = scramble_2x2_projected(depth, rng)
        cubes.append(cube)
        scramble_actions.append(actions)

    solved_step = np.full(episodes, -1, dtype=np.int32)
    first_actions = np.full(episodes, -1, dtype=np.int32)
    state = policy.initial_state(episodes, "cpu")

    with torch.no_grad():
        for step in range(max_steps):
            obs = encode_obs(cubes)
            logits, _, state = policy.forward_eval(obs, state)
            if sampled:
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(1).cpu().numpy()
            else:
                actions = torch.argmax(logits, dim=-1).cpu().numpy()

            for env_id, action in enumerate(actions):
                if solved_step[env_id] >= 0:
                    continue
                if first_actions[env_id] < 0:
                    first_actions[env_id] = int(action)
                cubes[env_id] = turn_2x2_projected(cubes[env_id], int(action))
                if is_solved(cubes[env_id]):
                    solved_step[env_id] = step + 1

            if np.all(solved_step >= 0):
                break

    solved = solved_step >= 0
    distances = np.array([corner_distance(cube) for cube in cubes], dtype=np.float32)
    avg_solve_steps = float(np.mean(solved_step[solved])) if np.any(solved) else 0.0
    return {
        "solve_rate": float(np.mean(solved)),
        "avg_solve_steps": avg_solve_steps,
        "avg_final_corner_distance": float(np.mean(distances)),
        "median_final_corner_distance": float(np.median(distances)),
        "first_action_mode": int(np.bincount(first_actions[first_actions >= 0], minlength=ACTIONS).argmax()),
    }


def parse_depths(value):
    depths = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            depths.extend(range(int(lo), int(hi) + 1))
        elif part:
            depths.append(int(part))
    return depths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="downloads/rubik_depth20_probe/best_depth20_probe.bin")
    parser.add_argument("--depths", default="1-14")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampled", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    policy = make_policy(checkpoint, args.hidden_size, args.num_layers)
    rng = np.random.default_rng(args.seed)
    depths = parse_depths(args.depths)

    print(f"checkpoint={checkpoint}")
    print(f"episodes={args.episodes} max_steps={args.max_steps} sampled={args.sampled}")
    print("depth solve_rate avg_solve_steps avg_corner_dist median_corner_dist first_action_mode")
    for depth in depths:
        result = evaluate_depth(policy, depth, args.episodes, args.max_steps, rng, args.sampled)
        print(
            f"{depth:5d} "
            f"{result['solve_rate']:.3f} "
            f"{result['avg_solve_steps']:.2f} "
            f"{result['avg_final_corner_distance']:.2f} "
            f"{result['median_final_corner_distance']:.1f} "
            f"{result['first_action_mode']:d}"
        )


if __name__ == "__main__":
    main()
