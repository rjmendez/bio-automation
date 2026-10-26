"""Quick sanity probe: run noise_det-v1 / liting_det-v1 / gendet-v1 on synthetic
images (clean vs. noisy, dark vs. bright) via onnxruntime CPU, to see whether
their outputs move in the expected direction. Not a benchmark -- just exploratory
signal-sanity-check per the research task.
"""
import numpy as np
import onnxruntime as ort

NOISE_PATH = r"C:\TopazModels\noise_det-v1-fp16-128x128-1x-ox\noise_det-v1-fp16.onnx"
LITING_PATH = r"C:\TopazModels\liting_det-v1-fp16-128x128-ox\liting_det-v1-fp16.onnx"
GENDET_PATH = r"C:\TopazModels\gendet-v1-fp32-256x256-1x-ox\gendet-v1-fp32-256x256-1x.onnx"

rng = np.random.default_rng(0)


def synth(h, w, base=0.5, noise_std=0.0, dtype=np.float32):
    img = np.full((h, w, 3), base, dtype=np.float32)
    # simple structured content so it isn't flat (gradient + a block)
    yy, xx = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img[..., 0] += 0.2 * xx
    img[..., 1] += 0.2 * yy
    img[20:60, 20:60, :] += 0.3
    if noise_std > 0:
        img += rng.normal(0, noise_std, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1).astype(dtype)
    return img[None, ...]  # NHWC batch=1


def run(path, inp_name, img):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    (out_name,) = [o.name for o in sess.get_outputs()]
    (in_info,) = sess.get_inputs()
    elem_type = in_info.type
    arr = img.astype(np.float16) if "float16" in elem_type else img.astype(np.float32)
    result = sess.run([out_name], {in_info.name: arr})[0]
    return result


print("=== noise_det-v1 ===")
for std, label in [(0.0, "clean"), (0.03, "light-noise"), (0.12, "heavy-noise")]:
    img = synth(128, 128, noise_std=std)
    out = run(NOISE_PATH, "patch0", img)
    print(f"  {label:12s} std={std:<5} -> noise_score={out.squeeze():.4f}")

print("=== liting_det-v1 ===")
for base, label in [(0.5, "mid-exposure"), (0.08, "dark/underexposed"), (0.92, "bright/overexposed")]:
    img = synth(128, 128, base=base)
    out = run(LITING_PATH, "image0", img)
    print(f"  {label:20s} base={base:<5} -> lighting_scores={np.array(out).squeeze()}")

print("=== gendet-v1 ===")
for std, label in [(0.0, "clean-synthetic"), (0.12, "noisy-synthetic")]:
    img = synth(256, 256, noise_std=std)
    out = run(GENDET_PATH, "image", img)
    logits = np.array(out).squeeze()
    sm = np.exp(logits - logits.max())
    sm = sm / sm.sum()
    top3 = np.argsort(-sm)[:3]
    print(f"  {label:18s} -> top3 class idx={top3.tolist()} probs={sm[top3].round(3).tolist()}")
