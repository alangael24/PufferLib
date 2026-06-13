import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

import pufferlib.pufferl as pufferl
from pufferlib import _C, torch_pufferl
from tools.render_rubik_policy import configure_depth20, native_bin_to_torch_state, choose_action


U, R, F, D, L, B = range(6)
FACE_STICKERS = 9
STICKERS = 54
COLORS = {
    0: (245, 245, 245),
    1: (190, 32, 38),
    2: (26, 150, 65),
    3: (245, 205, 38),
    4: (237, 125, 49),
    5: (35, 91, 168),
}


def load_args(depth, max_steps):
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        args = pufferl.load_config("rubik")
    finally:
        sys.argv = argv
    args = configure_depth20(args, depth=depth, max_steps=max_steps)
    args["vec"]["total_agents"] = 1
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 1
    args["train"]["horizon"] = 1
    args["train"]["minibatch_size"] = 1
    args["train"]["total_timesteps"] = 1
    return args


def decode_stickers(vec, env_id):
    obs = torch_pufferl._cpu_tensor(
        vec.obs_ptr, (vec.total_agents, vec.obs_size), torch.uint8)[env_id]
    return obs.view(STICKERS, 6).argmax(dim=1).cpu().numpy().astype(int).tolist()


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


def turn(stickers, action):
    axis, layer, turns = action_to_layer(action)
    out = stickers[:]
    for idx in range(STICKERS):
        x, y, z, nx, ny, nz = sticker_to_coord(idx)
        coord = x if axis == 0 else y if axis == 1 else z
        if coord != layer:
            continue
        x, y, z = rotate_vec(axis, turns, x, y, z)
        nx, ny, nz = rotate_vec(axis, turns, nx, ny, nz)
        out[coord_to_sticker(x, y, z, nx, ny, nz)] = stickers[idx]
    return out


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def scale(a, s):
    return a[0] * s, a[1] * s


def visible_index(face, row, col):
    if face == U:
        row = 2 - row
    return face * FACE_STICKERS + row * 3 + col


def draw_face(draw, stickers, face, origin, col_axis, row_axis):
    edge = (8, 18, 22)
    inset_col = scale(col_axis, 0.04)
    inset_row = scale(row_axis, 0.04)
    cell_col = scale(col_axis, 0.92)
    cell_row = scale(row_axis, 0.92)
    for row in range(3):
        for col in range(3):
            p = add(origin, add(scale(col_axis, col), scale(row_axis, row)))
            p = add(p, add(inset_col, inset_row))
            a = p
            b = add(p, cell_col)
            c = add(add(p, cell_col), cell_row)
            d = add(p, cell_row)
            color = COLORS[stickers[visible_index(face, row, col)]]
            draw.polygon([a, b, c, d], fill=color, outline=edge)


def draw_cube(stickers, step, max_steps, depth, solved, env_id):
    img = Image.new("RGB", (960, 720), (7, 13, 18))
    draw = ImageDraw.Draw(img)
    front = (280.0, 250.0)
    x_axis = (66.0, 0.0)
    y_axis = (0.0, 66.0)
    z_axis = (42.0, -32.0)
    draw_face(draw, stickers, U, front, x_axis, z_axis)
    draw_face(draw, stickers, R, add(front, scale(x_axis, 3.0)), z_axis, y_axis)
    draw_face(draw, stickers, F, front, x_axis, y_axis)
    label = f"depth {depth}   env {env_id}   step {step}/{max_steps}"
    if solved:
        label += "   solved"
    draw.text((24, 24), label, fill=(240, 245, 248))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="downloads/rubik_depth20_probe/best_depth20_probe.bin")
    parser.add_argument("--output", default="artifacts/rubik_depth20_policy.gif")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args_cli = parser.parse_args()

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    args = load_args(args_cli.depth, args_cli.max_steps)
    args["vec"]["total_agents"] = args_cli.agents
    if args_cli.env_id < 0 or args_cli.env_id >= args_cli.agents:
        raise ValueError("--env-id must be in [0, agents)")
    vec = _C.create_vec(args, 0)
    vec.reset()
    vec.log()
    policy = torch_pufferl.load_policy(args, vec)
    native_bin_to_torch_state(args_cli.checkpoint, policy)
    policy.eval()
    torch.manual_seed(args_cli.seed)
    state = policy.initial_state(vec.total_agents, "cpu")

    frames = []
    solved = False
    try:
        for step in range(args_cli.max_steps + 1):
            stickers = decode_stickers(vec, args_cli.env_id)
            frames.append(draw_cube(
                stickers, step, args_cli.max_steps, args_cli.depth, solved, args_cli.env_id))
            if step == args_cli.max_steps:
                break

            obs = torch_pufferl._cpu_tensor(
                vec.obs_ptr, (vec.total_agents, vec.obs_size), torch.uint8)
            with torch.no_grad():
                logits, _, state = policy.forward_eval(obs, state)
                action = choose_action(logits, args_cli.sampled)
            action_id = int(action[args_cli.env_id, 0].item())
            next_stickers = turn(stickers, action_id)
            vec.cpu_step(action.to(dtype=torch.float32).contiguous().data_ptr())
            terminals = torch_pufferl._cpu_tensor(
                vec.terminals_ptr, (vec.total_agents,), torch.float32)
            terminal = terminals[args_cli.env_id].item() > 0
            if terminal:
                solved = step + 1 < args_cli.max_steps
                if solved:
                    next_stickers = [face for face in range(6) for _ in range(9)]
                frames.append(draw_cube(
                    next_stickers, step + 1, args_cli.max_steps, args_cli.depth, solved,
                    args_cli.env_id))
                for _ in range(5):
                    frames.append(frames[-1])
                break
    finally:
        vec.close()

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=args_cli.duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"saved {output} frames={len(frames)} solved={solved}")


if __name__ == "__main__":
    main()
