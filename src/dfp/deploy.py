import argparse
import gc
import os
import sys
from typing import Tuple, List, Union, Dict, Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .data import convert_one_hot_to_image
from .net import deepfloorplanModel
from .net_func import deepfloorplanFunc
from .utils.rgb_ind_convertor import (
    floorplan_boundary_map,
    floorplan_fuse_map,
    ind2rgb,
)
from .utils.settings import overwrite_args_with_toml
from .utils.util import fill_break_line, flood_fill, refine_room_region

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def init(
    config: argparse.Namespace,
) -> Tuple[tf.keras.Model, tf.Tensor, np.ndarray]:
    if config.tfmodel == "subclass":
        model = deepfloorplanModel(config=config)
    elif config.tfmodel == "func":
        model = deepfloorplanFunc(config=config)
    # ↓↓↓ ここを書き換え ↓↓↓
    if config.loadmethod == "log":
        w = config.weight
        # ディレクトリ or 拡張子なし → TF Checkpoint とみなして復元
        if os.path.isdir(w) or os.path.splitext(w)[1] == "":
            ckpt_path = tf.train.latest_checkpoint(w) if os.path.isdir(w) else w
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint found under: {w}")
            tf.train.Checkpoint(model=model).restore(ckpt_path).expect_partial()
        else:
            # .weights.h5 / .h5 のファイルなら従来通り
            model.load_weights(w)
    elif config.loadmethod == "pb":
        # 既存の処理（必要なら tflite に切替えてもOK）
        model = tf.keras.models.load_model(config.weight)
    elif config.loadmethod == "tflite":
        # 既存の tflite 分岐
        pass
    img = mpimg.imread(config.image)[:, :, :3]
    shp = img.shape
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    img = tf.image.resize(img, [512, 512])
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [-1, 512, 512, 3])
    if tf.math.reduce_max(img) > 1.0:
        img /= 255
    if config.loadmethod == "tflite":
        return model, img, shp
    model.trainable = False
    if config.tfmodel == "subclass":
        model.vgg16.trainable = False
    return model, img, shp

def predict(model: tf.keras.Model,
            img: tf.Tensor,
            shp: Any) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Run inference with a Keras 3 model (subclass) without iterating layers manually.

    Notes
    -----
    * Keras 3 では InputLayer を layer(x) のように直接呼ぶと TypeError になるため、
      逐次ループではなく model(img, training=False) を使う。
    * 本リポジトリの `main()` 実装では、`--loadmethod log` のとき
        logits_cw, logits_r = predict(...)
      と受け取る一方、`--loadmethod pb/none` のときは
        logits_r, logits_cw = model(img)
      という順序になる。
      そのためここでは「モデルの生の出力（通常は (logits_r, logits_cw) ）」を受け取り、
      `(logits_cw, logits_r)` の順に並べ替えて返す。
    """
    # ---- 推論を一発で実行（Keras 3 互換）----
    outputs: Union[Tuple[tf.Tensor, tf.Tensor], List[tf.Tensor], Dict[str, tf.Tensor], tf.Tensor]
    outputs = model(img, training=False)

    logits_r: tf.Tensor = None
    logits_cw: tf.Tensor = None

    # 代表的な3パターンに対応
    if isinstance(outputs, (tuple, list)):
        if len(outputs) != 2:
            raise TypeError(f"Unexpected number of outputs: {len(outputs)} (expected 2).")
        # 既存コードの pb/none 分岐に合わせ、(r, cw) を想定
        logits_r, logits_cw = outputs[0], outputs[1]
    elif isinstance(outputs, dict):
        # キー名は環境により異なる可能性があるため候補を広めに見る
        # room/rooms/rt 等 → room タスクのロジット
        # cw/boundary/edges 等 → corner+wall（境界）タスクのロジット
        room_keys = ("room", "rooms", "rt", "logits_r", "room_logits")
        cw_keys   = ("cw", "boundary", "boundaries", "edges", "logits_cw", "cw_logits")
        for k in room_keys:
            if k in outputs:
                logits_r = outputs[k]
                break
        for k in cw_keys:
            if k in outputs:
                logits_cw = outputs[k]
                break
        if logits_r is None or logits_cw is None:
            raise KeyError(
                f"Cannot find expected keys in model outputs. Got keys: {list(outputs.keys())}"
            )
    else:
        # 単一テンソルが返る場合は本実装の前提に合わない
        raise TypeError(
            "Model returned a single tensor; expected a tuple/list/dict of two tensors (room, cw)."
        )

    # ---- 返却順（log 分岐の期待）に合わせる： (cw, r) ----
    return logits_cw, logits_r

def post_process(
    rm_ind: np.ndarray, bd_ind: np.ndarray, shp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    hard_c = (bd_ind > 0).astype(np.uint8)
    # region from room prediction
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)
    cw_mask = np.reshape(cw_mask, (*shp[:2], -1))
    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask, rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask.reshape(*shp[:2], -1) * new_rm_ind
    new_bd_ind = fill_break_line(bd_ind).squeeze()
    return new_rm_ind, new_bd_ind


def colorize(r: np.ndarray, cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cr = ind2rgb(r, color_map=floorplan_fuse_map)
    ccw = ind2rgb(cw, color_map=floorplan_boundary_map)
    return cr, ccw


def main(config: argparse.Namespace) -> np.ndarray:
    model, img, shp = init(config)
    if config.loadmethod == "tflite":
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]["index"], img)
        model.invoke()
        ri, cwi = 0, 1
        if config.tfmodel == "func":
            ri, cwi = 1, 0
        logits_r = model.get_tensor(output_details[ri]["index"])
        logits_cw = model.get_tensor(output_details[cwi]["index"])
        logits_cw = tf.convert_to_tensor(logits_cw)
        logits_r = tf.convert_to_tensor(logits_r)
    else:
        if config.tfmodel == "func":
            logits_r, logits_cw = model.predict(img)
        elif config.tfmodel == "subclass":
            if config.loadmethod == "log":
                logits_cw, logits_r = predict(model, img, shp)
            elif config.loadmethod == "pb" or config.loadmethod == "none":
                logits_r, logits_cw = model(img)
    logits_r = tf.image.resize(logits_r, shp[:2])
    logits_cw = tf.image.resize(logits_cw, shp[:2])
    r = convert_one_hot_to_image(logits_r)[0].numpy()
    cw = convert_one_hot_to_image(logits_cw)[0].numpy()

    if not config.colorize and not config.postprocess:
        cw[cw == 1] = 9
        cw[cw == 2] = 10
        r[cw != 0] = 0
        return (r + cw).squeeze()
    elif config.colorize and not config.postprocess:
        r_color, cw_color = colorize(r.squeeze(), cw.squeeze())
        return r_color + cw_color

    newr, newcw = post_process(r, cw, shp)
    if not config.colorize and config.postprocess:
        newcw[newcw == 1] = 9
        newcw[newcw == 2] = 10
        newr[newcw != 0] = 0
        return newr.squeeze() + newcw
    newr_color, newcw_color = colorize(newr.squeeze(), newcw.squeeze())
    result = newr_color + newcw_color
    print(shp, result.shape)

    if config.save:
        mpimg.imsave(config.save, result.astype(np.uint8))

    return result


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tfmodel", type=str, default="subclass", choices=["subclass", "func"]
    )
    p.add_argument("--image", type=str, default="resources/30939153.jpg")
    p.add_argument("--weight", type=str, default="log/store/G")
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--colorize", action="store_true")
    p.add_argument(
        "--loadmethod",
        type=str,
        default="log",
        choices=["log", "tflite", "pb", "none"],
    )  # log,tflite,pb
    p.add_argument("--save", type=str)
    p.add_argument(
        "--feature-channels",
        type=int,
        action="store",
        default=[256, 128, 64, 32],
        nargs=4,
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="vgg16",
        choices=["vgg16", "resnet50", "mobilenetv1", "mobilenetv2"],
    )
    p.add_argument(
        "--feature-names",
        type=str,
        action="store",
        nargs=5,
        default=[
            "block1_pool",
            "block2_pool",
            "block3_pool",
            "block4_pool",
            "block5_pool",
        ],
    )
    p.add_argument("--tomlfile", type=str, default=None)
    return p.parse_args(args)


def deploy_plot_res(result: np.ndarray):
    print(result.shape)
    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

def _assert_and_fix_order(logits_a, logits_b):
    """
    2つのテンソルから (logits_cw[... ,3], logits_r[... ,9]) の順で返す。
    どちらがどちらか不明でも自動判定する。
    """
    def _last(t):
        s = tuple(t.shape)
        return s[-1] if len(s) >= 3 else None

    ca, cb = _last(logits_a), _last(logits_b)
    # 期待：cw=3, r=9
    if (ca, cb) == (3, 9):
        return logits_a, logits_b
    if (ca, cb) == (9, 3):
        return logits_b, logits_a

    # どちらかが 3 か 9 なら判別
    if ca == 3:  # a が cw
        return logits_a, logits_b
    if cb == 3:  # b が cw
        return logits_b, logits_a
    if ca == 9:  # a が r
        return logits_b, logits_a
    if cb == 9:  # b が r
        return logits_a, logits_b

    raise ValueError(f"Unexpected channel sizes: {ca=}, {cb=}. Expected (3,9) in some order.")

def postprocess_logits(logits_cw, logits_r, thr_cw: float = 0.35):
    """
    logits を確率に変換してマップを作る。
    - cw はマルチラベル境界（3ch）想定 → sigmoid + しきい値
    - r は相互排他な部屋クラス（9ch）想定 → softmax + argmax
    返り値:
      room_map: (H,W) uint8  … 0..8 のクラスID
      cw_mask:  (H,W,3) uint8 … 各チャネルの0/1マスク
    """
    # 形を [H,W,C] に揃える
    if len(logits_cw.shape) == 4 and logits_cw.shape[0] == 1:
        logits_cw = logits_cw[0]
    if len(logits_r.shape) == 4 and logits_r.shape[0] == 1:
        logits_r = logits_r[0]

    # 確率化
    cw_prob = tf.sigmoid(logits_cw).numpy()         # (H,W,3)
    r_prob  = tf.nn.softmax(logits_r, axis=-1).numpy()  # (H,W,9)

    # マップ化
    cw_mask  = (cw_prob > thr_cw).astype(np.uint8)      # (H,W,3)
    room_map = np.argmax(r_prob, axis=-1).astype(np.uint8)  # (H,W)

    return room_map, cw_mask

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = overwrite_args_with_toml(args)
    result = main(args)
    deploy_plot_res(result)
    plt.show()
