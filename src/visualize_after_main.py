# -*- coding: utf-8 -*-
"""
TF2DeepFloorplan: result = main(args) の返却物を安全に可視化するユーティリティ
- 軸ずれ補正（HWC/CHW）
- 画像/ラベルの適切な補間（bilinear / nearest）
- ロジット or one-hot or ラベル(int) すべて自動対応
- result が dict/tuple どちらでも主要パターンを自動抽出
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# 基本ユーティリティ
# =========================

def _guess_layout_to_hwc(x: np.ndarray) -> np.ndarray:
    """2D/3D配列をHWCに正規化。グレースケール2Dはそのまま。"""
    a = np.asarray(x)
    if a.ndim == 2:
        return a
    if a.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape {a.shape}")
    H, W, C = a.shape
    # Cが小さいならHWC、先頭が小さい(<=4)ならCHWとみなす
    if C <= 4:
        return a
    if a.shape[0] <= 4:  # CHW想定
        return np.transpose(a, (1, 2, 0))
    return a  # 消極的にH/W/Cを維持

def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """float 0..1 / float 0..255 / int を安全に uint8 に整形"""
    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.floating):
        if a.max() <= 1.0:
            a = a * 255.0
        a = np.round(a)
    a = np.clip(a, 0, 255).astype(np.uint8)
    # グレースケール2DはH,Wのまま；3ch未満は3ch化
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] == 1:
        return np.repeat(a, 3, axis=2)
    if a.ndim == 3 and a.shape[2] == 2:
        # 2chは苦しいので最後をゼロ埋め
        z = np.zeros_like(a[..., :1])
        return np.concatenate([a, z], axis=2)
    return a

def _resize_image_bilinear(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """画像用BILINEAR"""
    a = _guess_layout_to_hwc(img)
    a = _to_uint8_image(a)
    pil = Image.fromarray(a)
    out = pil.resize((hw[1], hw[0]), resample=Image.BILINEAR)
    return np.asarray(out)

def _resize_label_nearest(lbl: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """整数ラベル（H,W）をNEARESTで"""
    m = np.asarray(lbl)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    pil = Image.fromarray(m.astype(np.int32), mode="I")
    out = pil.resize((hw[1], hw[0]), resample=Image.NEAREST)
    return np.asarray(out).astype(np.int32)

def _argmax_label_from_logits(x: tf.Tensor, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    ロジット/one-hotから (H,W) intラベルに。
    入力は (H,W,C) / (N,H,W,C) / (N,C,H,W) / (C,H,W) / (H,W) を許容。
    """
    t = tf.convert_to_tensor(x)
    # 形を統一してから解像度合わせ
    if t.ndim == 4 and t.shape[-1] <= 64:        # N,H,W,C
        pass
    elif t.ndim == 4 and t.shape[1] <= 64:       # N,C,H,W -> N,H,W,C
        t = tf.transpose(t, [0, 2, 3, 1])
    elif t.ndim == 3 and t.shape[-1] <= 64:      # H,W,C
        t = t[None, ...]
    elif t.ndim == 3 and t.shape[0] <= 64:       # C,H,W
        t = tf.transpose(t, [1, 2, 0])[None, ...]
    elif t.ndim == 2:                             # H,W （すでにラベルかも）
        return t.numpy().astype(np.int32)
    else:
        raise ValueError(f"Unexpected logits/label shape: {t.shape}")

    # ロジットのままBILINEARでサイズ合わせ → argmax
    t = tf.image.resize(t, size=out_hw, method="bilinear")
    lab = tf.argmax(t, axis=-1)  # (N,H,W)
    return tf.cast(lab, tf.int32).numpy()[0]

def _make_palette(num_classes: int) -> np.ndarray:
    base = np.array([
        [0, 0, 0],        # 0: 背景
        [255, 99, 132],   # 1
        [54, 162, 235],   # 2
        [255, 206, 86],   # 3
        [75, 192, 192],   # 4
        [153, 102, 255],  # 5
        [255, 159, 64],   # 6
    ], dtype=np.uint8)
    if num_classes <= len(base):
        return base[:num_classes]
    # 追加色を乱数で（固定シード）
    rng = np.random.default_rng(0)
    extra = rng.integers(0, 256, size=(num_classes - len(base), 3), dtype=np.uint8)
    return np.vstack([base, extra])

def _colorize_label(lab: np.ndarray, palette: Optional[np.ndarray] = None) -> np.ndarray:
    lab = np.asarray(lab).astype(np.int32)
    K = int(lab.max()) + 1 if lab.size else 1
    palette = _make_palette(K) if palette is None else palette
    h, w = lab.shape
    out = np.zeros((h, w, 3), np.uint8)
    for k in range(K):
        out[lab == k] = palette[k]
    return out

def _overlay(base_rgb: np.ndarray, color_mask_rgb: np.ndarray, alpha: float=0.45) -> np.ndarray:
    base = _to_uint8_image(base_rgb)
    mask = _to_uint8_image(color_mask_rgb)
    base = base if base.ndim == 3 else np.stack([base]*3, axis=2)
    mask = mask if mask.ndim == 3 else np.stack([mask]*3, axis=2)
    out = (base.astype(np.float32) * (1 - alpha) + mask.astype(np.float32) * alpha).round()
    return np.clip(out, 0, 255).astype(np.uint8)

# =========================
# result から安全に取り出す
# =========================

def _first_key(d: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None

def _extract_from_result(result: Union[Dict[str, Any], Tuple, list]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Tuple[int,int]]:
    """
    result から (img, logits_or_label_r, logits_or_label_cw, (H,W)) を取り出す。
    返るラベルはロジット/onehot/ラベル（どれでもOKの“中間”）で、可視化側で最終処理。
    """
    img = None
    r_map = None
    cw_map = None
    out_hw = None

    if isinstance(result, dict):
        # 画像
        k_img = _first_key(result, ["img", "image", "input", "inp", "x", "img_rgb"])
        if k_img is not None:
            img = np.asarray(result[k_img])

        # 形
        k_shp = _first_key(result, ["shp", "shape", "size", "target_shape", "out_shape"])
        if k_shp is not None:
            shp = result[k_shp]
            # (H,W,...) or (H,W)
            if isinstance(shp, (list, tuple, np.ndarray)) and len(shp) >= 2:
                out_hw = (int(shp[0]), int(shp[1]))

        # ルーム
        for key in ["logits_r", "room_logits", "pred_r", "r_logits", "r", "rooms", "room", "label_r", "mask_r"]:
            if key in result:
                r_map = np.asarray(result[key]); break

        # 壁/輪郭
        for key in ["logits_cw", "contour_logits", "wall_logits", "pred_cw", "cw_logits", "cw", "contour", "wall", "label_cw", "mask_cw"]:
            if key in result:
                cw_map = np.asarray(result[key]); break

    elif isinstance(result, (tuple, list)):
        # よくある並び: (model, img, shp) / (img, logits_cw, logits_r, shp) 等に対応
        # ヒューリスティック：画像（HWC/CHW, 3ch or 1chっぽい）と shp(2~3長) を推定
        cand = [np.asarray(x) if isinstance(x, (np.ndarray, tf.Tensor)) else x for x in result]

        # shp 候補
        for x in cand:
            if isinstance(x, (list, tuple, np.ndarray)) and not hasattr(x, "dtype"):
                if len(x) >= 2 and all(isinstance(int(v), (int, np.integer)) for v in [x[0], x[1]]):
                    out_hw = (int(x[0]), int(x[1]))
                    break

        # 画像候補
        for x in cand:
            if isinstance(x, np.ndarray) and x.ndim in (2,3) and (x.ndim==3 and min(x.shape[0], x.shape[-1])<=4):
                img = x
                break

        # 2つのマップ候補（残りの配列）
        rest = [x for x in cand if isinstance(x, np.ndarray) and x is not img]
        # logits/onehot/label のいずれか2つを拾う
        if len(rest) >= 2:
            # チャンネル数や値域を見てそれっぽく割当て
            a, b = rest[0], rest[1]
            r_map, cw_map = a, b
        elif len(rest) == 1:
            # どちらか片方のみ
            r_map = rest[0]

    # out_hw 未決定なら、img か マップ から決定
    if out_hw is None:
        if img is not None:
            a = _guess_layout_to_hwc(img)
            out_hw = (a.shape[0], a.shape[1])
        elif r_map is not None and r_map.ndim >= 2:
            if r_map.ndim == 2:
                out_hw = (r_map.shape[0], r_map.shape[1])
            elif r_map.ndim == 3:
                # HWC or CHW か NHCW の可能性があるが、とりあえず2軸を拾う
                out_hw = (int(r_map.shape[-3] if r_map.shape[-1] <= 64 else r_map.shape[-2]),
                          int(r_map.shape[-2] if r_map.shape[-1] <= 64 else r_map.shape[-1]))
        else:
            raise ValueError("出力サイズ(out_hw)を決められませんでした。resultに img/shp/予測が含まれているか確認してください。")

    return img, r_map, cw_map, out_hw

# =========================
# メイン可視化関数
# =========================

def visualize_from_result(
    result: Union[Dict[str, Any], Tuple, list],
    save_path: Optional[str] = "viz_result.png",
    save_overlay_path: Optional[str] = "viz_overlay.png",
    palette: Optional[np.ndarray] = None,
    show: bool = True,
) -> Dict[str, np.ndarray]:
    """
    result = main(args) の返却物を可視化。
    - 返り値: {"input":RGB, "rooms":RGB, "cw":RGB, "overlay_rooms":RGB, "overlay_cw":RGB}
    """
    img, r_map, cw_map, out_hw = _extract_from_result(result)

    # 入力画像の整形
    if img is None:
        # 入力画像が無い場合は真っ白画像をベースにする
        base = np.full((out_hw[0], out_hw[1], 3), 255, np.uint8)
    else:
        base = _resize_image_bilinear(img, out_hw)

    # Room
    vis_r = None
    if r_map is not None:
        r_lab = _argmax_label_from_logits(r_map, out_hw)  # (H,W) int
        vis_r = _colorize_label(r_lab, palette)

    # Contour/Wall
    vis_cw = None
    if cw_map is not None:
        cw_lab = _argmax_label_from_logits(cw_map, out_hw)
        vis_cw = _colorize_label(cw_lab, palette)

    # 図をまとめて描画
    cols = 1 + int(vis_r is not None) + int(vis_cw is not None)
    plt.figure(figsize=(4*cols, 4), dpi=120)
    col = 1
    plt.subplot(1, cols, col); col += 1
    plt.imshow(base); plt.title("Input"); plt.axis("off")

    if vis_r is not None:
        plt.subplot(1, cols, col); col += 1
        plt.imshow(vis_r); plt.title("Rooms"); plt.axis("off")

    if vis_cw is not None:
        plt.subplot(1, cols, col); col += 1
        plt.imshow(vis_cw); plt.title("Contour/Wall"); plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    if show:
        plt.show()

    # オーバレイ画像も任意で保存
    outputs = {"input": base}
    if vis_r is not None:
        over_r = _overlay(base, vis_r, alpha=0.45)
        outputs["rooms"] = vis_r
        outputs["overlay_rooms"] = over_r
    if vis_cw is not None:
        over_cw = _overlay(base, vis_cw, alpha=0.45)
        outputs["cw"] = vis_cw
        outputs["overlay_cw"] = over_cw

    if save_overlay_path:
        # overlayが2つある場合は横並びで1枚に
        if "overlay_rooms" in outputs and "overlay_cw" in outputs:
            h, w, _ = base.shape
            canvas = np.zeros((h, w*2, 3), np.uint8)
            canvas[:, :w] = outputs["overlay_rooms"]
            canvas[:, w:] = outputs["overlay_cw"]
            Image.fromarray(canvas).save(save_overlay_path)
        elif "overlay_rooms" in outputs:
            Image.fromarray(outputs["overlay_rooms"]).save(save_overlay_path)
        elif "overlay_cw" in outputs:
            Image.fromarray(outputs["overlay_cw"]).save(save_overlay_path)
        # どちらも無ければ何もしない

    return outputs


# =========================
# 使い方（例）
# =========================
if __name__ == "__main__":
    """
    例）既存の main.py がある前提で:
        from main import main, get_args
        args = get_args()  # 既存の引数取得関数がある場合
        result = main(args)
        visualize_from_result(result, save_path="viz_result.png", save_overlay_path="viz_overlay.png")

    ここでは import だけ試み、なければスキップ（ノートブックから呼ぶ想定）
    """
    try:
        from main import main, get_args  # 既存実装があれば使う
        try:
            args = get_args()
        except Exception:
            # get_args が無い場合は、ダミーのNamespaceを与える（必要に応じて編集）
            from types import SimpleNamespace
            args = SimpleNamespace()
        result = main(args)
        visualize_from_result(result, save_path="viz_result.png", save_overlay_path="viz_overlay.png")
    except Exception as e:
        print("[INFO] 直接実行はスキップしました。ノートブック/他スクリプトから visualize_from_result(result) を呼んでください。")
        print("Reason:", repr(e))
