"""End-to-end nnU-Net v2 pipeline for an MSD 3D segmentation task.

Single entry point. After installing nnU-Net and extracting the MSD task:

    python pipeline.py                 # convert -> preprocess -> train -> evaluate
    python pipeline.py --no-convert    # skip stages already done
    python pipeline.py --no-train --no-preprocess --no-convert  # evaluate only

nnU-Net's three working directories are taken from config and exported as the
env vars it requires.
"""
import os
import glob
import json
import subprocess

import config


def make_env():
    for d in (config.NNUNET_RAW, config.NNUNET_PREPROCESSED, config.NNUNET_RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["nnUNet_raw"] = str(config.NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(config.NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(config.NNUNET_RESULTS)
    return env


def run(cmd, env):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def convert(env):
    run(["nnUNetv2_convert_MSD_dataset", "-i", str(config.MSD_EXTRACTED),
         "-overwrite_id", str(config.DATASET_ID)], env)


def preprocess(env):
    run(["nnUNetv2_plan_and_preprocess", "-d", str(config.DATASET_ID),
         "-np", str(config.NUM_PROC), "--verify_dataset_integrity"], env)


def train(env):
    cmd = ["nnUNetv2_train", str(config.DATASET_ID), config.CONFIG, str(config.FOLD)]
    if getattr(config, "TRAINER", None):
        cmd += ["-tr", config.TRAINER]
        cmd += ["--c"]
    run(cmd, env)


def evaluate(env):
    pattern = str(
        config.NNUNET_RESULTS / config.DATASET_NAME / f"*{config.CONFIG}*"
        / f"fold_{config.FOLD}" / "validation" / "summary.json"
    )
    hits = glob.glob(pattern)
    if not hits:
        print(f"No summary.json found under:\n  {pattern}\nHas training finished?")
        return
    with open(hits[0]) as f:
        s = json.load(f)
    dice = s.get("foreground_mean", {}).get("Dice")
    line = f"{config.DATASET_NAME} | {config.CONFIG} | fold {config.FOLD} | validation Dice (foreground mean): {dice}"
    print("\n" + line)
    out = config.PROJECT_ROOT / "nnunet_summary.txt"
    out.write_text(line + "\n")
    print(f"Saved to {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="nnU-Net v2 MSD pipeline")
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    env = make_env()
    if not args.no_convert:
        convert(env)
    if not args.no_preprocess:
        preprocess(env)
    if not args.no_train:
        train(env)
    if not args.no_eval:
        evaluate(env)
    print("\nObjective 2 pipeline complete.")


if __name__ == "__main__":
    main()
