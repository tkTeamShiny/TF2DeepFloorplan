# -*- coding: utf-8 -*-
"""
TF2DeepFloorplan: result = main(args) の返却物を安全に可視化するユーティリティ（改訂版）
- result: dict / tuple / list / 任意オブジェクト（属性） すべて対応
- 画像/ラベル補間: 画像=BILINEAR / ラベル=NEAREST
- ロジット or one-hot or 既にラベルのいずれにも対応（argmax含む）
- out_hw を args/result/予測配列 から堅牢に推定。必要なら引数 out_hw= で明示指定可能
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
    # HWC or CHW の簡易判定
    if a.shape[-1] <= 4:
        return a
    if a.shape[0] <= 4:
        return np.transpose(a, (1, 2, 0))
    return a

def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """float 0..1 / float 0..255 / int を安全に uint8 に整形"""
    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.floating):
        if a.size and np.nanmax(a) <= 1.0:
            a = a * 255.0
        a = np.round(a)
    a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] == 1:
        return np.repeat(a, 3, axis=2)
    if a.ndim == 3 and a.shape[2] == 2:
        z = np.zeros_like(a[..., :1])
        return np.concatenate([a, z], axis=2)
    return a

def _resize_image_bilinear(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    a = _guess_layout_to_hwc(img)
    a = _to_uint8_image(a)
    pil = Image.fromarray(a)
    out = pil.resize((hw[1], hw[0]), resample=Image.BILINEAR)
    return np.asarray(out)

def _resize_label_nearest(lbl: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    m = np.asarray(lbl)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    pil = Image.fromarray(m.astype(np.int32), mode="I")
    out = pil.resize((hw[1], hw[0]), resample=Image.NEAREST)
    return np.asarray(out).astype(np.int32)

def _shape_list(x) -> Optional[Sequence[Optional[int]]]:
    """shape を list[int or None] にする（tf.Tensor もOK）"""
    if hasattr(x, "shape"):
        s = x.shape
        try:
            if hasattr(s, "as_list"):
                s = s.as_list()
            else:
                s = list(s)
            return [None if v is None else int(v) for v in s]
        except Exception:
            return None
    return None

def _is_probably_channels(v: Optional[int]) -> bool:
    return v is not None and v <= 64  # クラス数はだいたい<=64を想定

def _infer_hw_from_array(x) -> Optional[Tuple[int, int]]:
    """
    ロジット/ラベル/画像配列の shape から (H,W) を推定。
    対応: 2D, 3D(HWC/CHW), 4D(NHWC/NCHW), 5D(…HW…)も後方2軸優先で推定。
    """
    s = _shape_list(x)
    if not s:
        return None

    # 2D: H,W
    if len(s) == 2 and all(isinstance(v, int) for v in s):
        return (s[0], s[1])

    # 3D
    if len(s) == 3:
        H, W, C = s[0], s[1], s[2]
        # HWC?
        if _is_probably_channels(C) and (H and W and H > 4 and W > 4):
            return (H, W)
        # CHW?
        C2, H2, W2 = s[0], s[1], s[2]
        if _is_probably_channels(C2) and (H2 and W2 and H2 > 4 and W2 > 4):
            return (H2, W2)
        # fallback: 大きい2軸をHWとみなす
        idx = np.argsort([v if v is not None else -1 for v in s])[-2:]
        idx = sorted(idx)
        if len(idx) == 2 and all(s[i] and s[i] > 4 for i in idx):
            return (s[idx[0]], s[idx[1]])
        return None

    # 4D: NHWC or NCHW
    if len(s) == 4:
        N, A, B, C = s
        # NHWC: (N,H,W,C)
        if _is_probably_channels(C) and (A and B and A > 4 and B > 4):
            return (A, B)
        # NCHW: (N,C,H,W)
        if _is_probably_channels(A) and (B and C and B > 4 and C > 4):
            return (B, C)
        # 他: 後方2軸がHWのことが多い
        if s[-2] and s[-1] and s[-2] > 4 and s[-1] > 4:
            return (s[-2], s[-1])
        return None

    # 5D 以上: 最後の2軸が HW であることがほとんど
    if len(s) >= 5 and s[-2] and s[-1] and s[-2] > 4 and s[-1] > 4:
        return (s[-2], s[-1])

    return None

def _argmax_label_from_logits(x, out_hw: Tuple[int, int]) -> np.ndarray:
    """ロジット/one-hot/ラベル(int) → (H,W) int"""
    # 既に 2D ラベルならそのまま
    s = _shape_list(x)
    if s and len(s) == 2:
        return np.asarray(x).astype(np.int32)

    t = tf.convert_to_tensor(x)
    # 正規化: 4Dは N,H,W,C / 3Dは H,W,C にする
    if t.ndim == 4 and t.shape[-1] is not None and t.shape[-1] <= 64:
        pass  # NHWC
    elif t.ndim == 4 and t.shape[1] is not None and t.shape[1] <= 64:
        t = tf.transpose(t, [0, 2, 3, 1])  # NCHW -> NHWC
    elif t.ndim == 3 and t.shape[-1] is not None and t.shape[-1] <= 64:
        t = t[None, ...]
    elif t.ndim == 3 and t.shape[0] is not None and t.shape[0] <= 64:
        t = tf.transpose(t, [1, 2, 0])[None, ...]  # CHW -> HWC (+N)
    elif t.ndim == 2:
        return t.numpy().astype(np.int32)
    else:
        # それでも合わない場合は最後の2軸がHWだと仮定
        if t.ndim >= 3:
            # N???HW? -> H,Wが最後の2つである前提で移動
            while t.ndim < 4:
                t = t[None, ...]
            # チャンネルが最後でない場合は移動（厳密には不明だが推定）
            if t.shape[-1] is None or t.shape[-1] > 64:
                # 適当に最後へ移す（ダミー1ch化）
                t = tf.expand_dims(t, axis=-1)
        else:
            raise ValueError(f"Unexpected shape for logits/label: {t.shape}")

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
# result / args から情報を拾う
# =========================

def _object_to_mapping(obj) -> Dict[str, Any]:
    """オブジェクトの属性を辞書化（callableや__xxx__は除外）"""
    d: Dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if callable(val):
            continue
        d[name] = val
    return d

def _first_key(d: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None

def _maybe_out_hw_from_args(args: Any) -> Optional[Tuple[int, int]]:
    """args から (H,W) を推測"""
    if args is None:
        return None
    d = args if isinstance(args, dict) else _object_to_mapping(args)
    # 代表的な名前の探索
    cand_pairs = [
        ("height","width"),
        ("img_h","img_w"),
        ("H","W"),
        ("h","w"),
    ]
    for a,b in cand_pairs:
        if a in d and b in d:
            try:
                return (int(d[a]), int(d[b]))
            except Exception:
                pass
    # 単一サイズ（正方形）
    for k in ["img_size","size","input_size","test_size","resize","resolution"]:
        if k in d:
            try:
                v = d[k]
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    return (int(v[0]), int(v[1]))
                return (int(v), int(v))
            except Exception:
                pass
    return None

def _extract_from_result(
    result: Union[Dict[str, Any], Tuple, list, object],
    args: Any = None
) -> Tuple[Optional[np.ndarray], Optional[Any], Optional[Any], Optional[Tuple[int,int]]]:
    """
    (img, r_map, cw_map, out_hw or None) を取り出す。
    out_hw は推定に失敗したら None を返す（上位でさらに推定/明示指定）。
    """
    img = None
    r_map = None
    cw_map = None
    out_hw = None

    # args から（もし渡っていれば）先に out_hw を候補として拾う
    out_hw = _maybe_out_hw_from_args(args)

    # result を dict 化っぽく扱う
    if isinstance(result, dict):
        d = result
    elif isinstance(result, (tuple, list)):
        d = {}  # 取り出しは後で
    else:
        d = _object_to_mapping(result)

    if isinstance(result, dict) or not isinstance(result, (tuple, list)):
        k_img = _first_key(d, ["img", "image", "input", "inp", "x", "img_rgb"])
        if k_img is not None:
            img = np.asarray(d[k_img])

        # 形の候補
        k_shp = _first_key(d, ["shp", "shape", "size", "target_shape", "out_shape", "test_shape"])
        if k_shp is not None:
            shp = d[k_shp]
            if isinstance(shp, (list, tuple, np.ndarray)) and len(shp) >= 2:
                out_hw = out_hw or (int(shp[0]), int(shp[1]))

        # ルーム
        for key in ["logits_r", "room_logits", "pred_r", "r_logits", "r", "rooms", "room", "label_r", "mask_r"]:
            if key in d:
                r_map = d[key]; break

        # 壁/輪郭
        for key in ["logits_cw", "contour_logits", "wall_logits", "pred_cw", "cw_logits", "cw", "contour", "wall", "label_cw", "mask_cw"]:
            if key in d:
                cw_map = d[key]; break

    if isinstance(result, (tuple, list)):
        # 代表的な並び: (img, logits_cw, logits_r, shp) 等
        items = list(result)
        # shp 候補
        for x in items:
            if isinstance(x, (list, tuple, np.ndarray)) and not hasattr(x, "dtype"):
                if len(x) >= 2:
                    try:
                        out_hw = out_hw or (int(x[0]), int(x[1]))
                        break
                    except Exception:
                        pass
        # 画像候補（HWC/CHW）
        for x in items:
            if isinstance(x, (np.ndarray, tf.Tensor)):
                sl = _shape_list(x)
                if sl and len(sl) in (2,3) and (len(sl)==2 or (len(sl)==3 and min(sl[0], sl[-1])<=4)):
                    img = np.asarray(x)
                    break
        # 予測候補
        rest = [x for x in items if isinstance(x, (np.ndarray, tf.Tensor)) and x is not img]
        if len(rest) >= 2:
            r_map, cw_map = rest[0], rest[1]
        elif len(rest) == 1:
            r_map = rest[0]

    # out_hw 未確定なら、img/予測から推定
    if out_hw is None:
        for cand in [img, r_map, cw_map]:
            if cand is not None:
                hw = _infer_hw_from_array(cand)
                if hw is not None:
                    out_hw = hw
                    break

    return (None if img is None else np.asarray(img)), r_map, cw_map, out_hw

# =========================
# メイン可視化関数
# =========================

def visualize_from_result(
    result: Union[Dict[str, Any], Tuple, list, object],
    save_path: Optional[str] = "viz_result.png",
    save_overlay_path: Optional[str] = "viz_overlay.png",
    palette: Optional[np.ndarray] = None,
    show: bool = True,
    *,
    args: Any = None,
    out_hw: Optional[Tuple[int,int]] = None,
) -> Dict[str, np.ndarray]:
    """
    result = main(args) の返却物を可視化。
    - args: main に渡した args（あればサイズ推定に使用）
    - out_hw: (H,W) を明示指定したい場合に指定
    - 返り値: {"input":RGB, "rooms":RGB, "cw":RGB, "overlay_rooms":RGB, "overlay_cw":RGB}
    """
    img, r_map, cw_map, guessed = _extract_from_result(result, args=args)
    out_hw_final = out_hw or guessed
    if out_hw_final is None:
        # 情報提供のため、手掛かりを列挙してからエラーを投げる
        shapes = {
            "img": _shape_list(img) if img is not None else None,
            "r_map": _shape_list(r_map) if r_map is not None else None,
            "cw_map": _shape_list(cw_map) if cw_map is not None else None,
            "args_hint": _maybe_out_hw_from_args(args),
        }
        raise ValueError(
            f"出力サイズ(out_hw)を決められませんでした。out_hw= で明示指定してください。\n"
            f"手掛かり shapes: {shapes}"
        )

    # 入力画像の整形
    if img is None:
        base = np.full((out_hw_final[0], out_hw_final[1], 3), 255, np.uint8)
    else:
        base = _resize_image_bilinear(img, out_hw_final)

    # Room
    vis_r = None
    if r_map is not None:
        r_lab = _argmax_label_from_logits(r_map, out_hw_final)
        vis_r = _colorize_label(r_lab, palette)

    # Contour/Wall
    vis_cw = None
    if cw_map is not None:
        cw_lab = _argmax_label_from_logits(cw_map, out_hw_final)
        vis_cw = _colorize_label(cw_lab, palette)

    # 描画
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

    # オーバレイ
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

    return outputs


# =========================
# 直接実行（任意）
# =========================
if __name__ == "__main__":
    try:
        from main import main, get_args  # 既存実装があれば使う
        try:
            args = get_args()
        except Exception:
            from types import SimpleNamespace
            args = SimpleNamespace()
        result = main(args)
        visualize_from_result(result, save_path="viz_result.png", save_overlay_path="viz_overlay.png", args=args)
    except Exception as e:
        print("[INFO] 直接実行はスキップ。ノートブック/別スクリプトから visualize_from_result(result, args=..., out_hw=...) を呼んでください。")
        print("Reason:", repr(e))
