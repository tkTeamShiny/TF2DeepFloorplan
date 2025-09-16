import argparse
import os
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# print('Is gpu available: ',tf.test.is_gpu_available());

import tensorflow as tf
import numpy as np

def make_const_kernel_initializer(kernel_2d_or_4d):
    """(kh,kw) もしくは (kh,kw,1,1) のカーネルを (kh,kw,in_ch,out_ch) にタイルして返す。"""
    k = tf.convert_to_tensor(kernel_2d_or_4d, dtype=tf.float32)
    if k.shape.rank == 2:  # (kh, kw) -> (kh, kw, 1, 1)
        k = tf.reshape(k, (k.shape[0], k.shape[1], 1, 1))
    def _init(shape, dtype=None):
        kh, kw, in_ch, out_ch = shape
        base = tf.reshape(k, (k.shape[0], k.shape[1], 1, 1))
        if base.shape[0] != kh or base.shape[1] != kw:
            base = tf.image.resize(base, (kh, kw), method="nearest")
        base = tf.tile(base, [1, 1, in_ch, out_ch])
        return tf.cast(base, dtype or tf.float32)
    return _init

def conv2d(
    dim: int,
    size: int = 3,
    stride: int = 1,
    rate: int = 1,
    pad: str = "same",
    act: str = "relu",
) -> tf.keras.Sequential:
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            dim, size, strides=stride, padding=pad, dilation_rate=rate
        )
    )
    if act == "leaky":
        result.add(tf.keras.layers.LeakyReLU())
    elif act == "relu":
        result.add(tf.keras.layers.ReLU())
    return result


def max_pool2d(
    size: int = 2, stride: int = 2, pad: str = "valid"
) -> tf.keras.Sequential:
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.MaxPool2D(pool_size=size, strides=stride, padding=pad)
    )
    return result


def upconv2d(
    dim: int,
    size: int = 4,
    stride: int = 2,
    pad: str = "same",
    act: str = "relu",
) -> tf.keras.Sequential:
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(dim, size, strides=stride, padding=pad)
    )
    if act == "relu":
        result.add(tf.keras.layers.ReLU())
    return result


def up_bilinear(dim: int) -> tf.keras.Sequential:
    result = tf.keras.Sequential()
    result.add(conv2d(dim, size=1, act="linear"))
    return result


class deepfloorplanModel(Model):
    def __init__(self, config: argparse.Namespace = None):
        super(deepfloorplanModel, self).__init__()
        self.config = config
        dimlist = [256, 128, 64, 32]
        self.feature_names = [
            "block1_pool",
            "block2_pool",
            "block3_pool",
            "block4_pool",
            "block5_pool",
        ]
        if config is not None:
            dimlist = config.feature_channels
            assert (
                config.backbone == "vgg16"
            ), "subclass backbone must be vgg16"
            self.feature_names = config.feature_names
        self._vgg16init()
        # room boundary prediction (rbp)
        self.rbpups = [upconv2d(dim=d, act="linear") for d in dimlist]
        self.rbpcv1 = [conv2d(dim=d, act="linear") for d in dimlist]
        self.rbpcv2 = [conv2d(dim=d) for d in dimlist]
        self.rbpfinal = up_bilinear(3)

        # room type prediction (rtp)
        self.rtpups = [upconv2d(dim=d, act="linear") for d in dimlist]
        self.rtpcv1 = [conv2d(dim=d, act="linear") for d in dimlist]
        self.rtpcv2 = [conv2d(dim=d) for d in dimlist]

        # attention map
        self.atts1 = [conv2d(dim=dimlist[i]) for i in range(len(dimlist))]
        self.atts2 = [conv2d(dim=dimlist[i]) for i in range(len(dimlist))]
        self.atts3 = [
            conv2d(dim=1, size=1, act="sigmoid") for i in range(len(dimlist))
        ]

        # reduce the tensor depth
        self.xs1 = [conv2d(dim=d) for d in dimlist]
        self.xs2 = [conv2d(dim=1, size=1, act="linear") for d in dimlist]

        # context conv2d
        dak = [9, 17, 33, 65]  # kernel_shape=[h,v,inc,outc]
        # horizontal
        self.hs = [self.constant_kernel((d, 1, 1, 1)) for d in dak]
        # 変更後（Keras 3 対応）
        self.hf = []
        for i, d in enumerate(dak):
            self.hf.append(
                tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(d, 1),
                    use_bias=False,
                    padding="same",
                    kernel_initializer=make_const_kernel_initializer(self.hs[i]),
                    trainable=False,
                    name=f"hf_{i}",
                )
            )
        # vertical
        self.vs = [self.constant_kernel((1, d, 1, 1)) for d in dak]
        # 変更後（OK）
        self.vf = []
        for i, d in enumerate(dak):
            self.vf.append(
                tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(1, d),
                    use_bias=False,
                    padding="same",
                    kernel_initializer=make_const_kernel_initializer(self.vs[i]),
                    trainable=False,
                    name=f"vf_{i}",
                )
            )
        self.ds = [self.constant_kernel((d, d, 1, 1), diag=True) for d in dak]
        self.df = []
        for i, d in enumerate(dak):
            self.df.append(
                tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(d, d) if isinstance(d, int) else tuple(d),
                    use_bias=False,
                    padding="same",
                    kernel_initializer=make_const_kernel_initializer(self.ds[i]),
                    trainable=False,
                    name=f"df_{i}",
                )
            )
        # diagonal flip
        self.dfs = [
            self.constant_kernel((d, d, 1, 1), diag=True, flip=True)
            for d in dak
        ]
        self.dff = []
        for i, d in enumerate(dak):
            self.dff.append(
                tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=(d, d) if isinstance(d, int) else tuple(d),
                    use_bias=False,
                    padding="same",
                    # ★ ここ：元々 weights=[・・・] に渡していた “同じカーネル変数” をそのまま渡す
                    kernel_initializer=make_const_kernel_initializer(self.dfs[i]),
                    trainable=False,
                    name=f"dff_{i}",
                )
            )

        # expand dim
        self.ed = [conv2d(dim=d, size=1, act="linear") for d in dimlist]
        # learn rich feature
        self.lrf = [conv2d(dim=d) for d in dimlist]
        # final
        self.rtpfinal = up_bilinear(9)

    def _vgg16init(self):
        self.vgg16 = VGG16(
            weights="imagenet", include_top=False, input_shape=(512, 512, 3)
        )
        for layer in self.vgg16.layers:
            layer.trainable = False

    def constant_kernel(
        self,
        shape: Tuple[int, int, int, int],
        val: int = 1,
        diag: bool = False,
        flip: bool = False,
    ) -> np.ndarray:
        k = np.array([]).astype(int)
        if not diag:
            k = val * np.ones(shape)
        else:
            w = np.eye(shape[0], shape[1])
            if flip:
                w = w.reshape((shape[0], shape[1], 1))
                w = np.flip(w, 1)
            k = w.reshape(shape)
        return k

    def non_local_context(
        self, t1: tf.Tensor, t2: tf.Tensor, idx: int, stride: int = 4
    ) -> tf.Tensor:
        N, H, W, C = t1.shape.as_list()
        hs = H // stride if (H // stride) > 1 else (stride - 1)
        vs = W // stride if (W // stride) > 1 else (stride - 1)
        hs = hs if (hs % 2 != 0) else hs + 1
        vs = hs if (vs % 2 != 0) else vs + 1
        a = t1
        x = t2
        a = self.atts1[idx](a)
        a = self.atts2[idx](a)
        a = self.atts3[idx](a)
        a = tf.keras.activations.sigmoid(a)
        x = self.xs1[idx](x)
        x = self.xs2[idx](x)
        x = a * x

        h = self.hf[idx](x)
        v = self.vf[idx](x)
        d = self.df[idx](x)
        f = self.dff[idx](x)
        c1 = a * (h + v + d + f)
        c1 = self.ed[idx](c1)

        features = tf.concat([t2, c1], axis=3)
        out = self.lrf[idx](features)
        return out

    def call(self, x: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Keras 3 互換の forward 実装（完全版）。
        - VGG16 の InputLayer は callable ではないためスキップ
        - その他の層は training 引数の有無に応じて安全に呼び出し
        - 特徴マップの回収は self.feature_names（未設定なら 'pool' 名）に従う
        - 以降の RBP/ RTP の処理は元実装と同一
        戻り値: (logits_r, logits_cw)
        """
        features = []
        feature = x
    
        # ====== VGG16 エンコーダ（InputLayer を呼ばない）======
        for layer in self.vgg16.layers:
            # Keras 3: InputLayer は layer(feature) できないのでスキップ
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
    
            # 一部の層（例: BatchNorm）は training を受け取る。
            # 受け取らない層もあるため try/except で安全に呼ぶ。
            try:
                feature = layer(feature, training=training)
            except TypeError:
                feature = layer(feature)
    
            # 特徴マップの回収
            name = getattr(layer, "name", "")
            if hasattr(self, "feature_names") and isinstance(self.feature_names, (list, tuple)) and self.feature_names:
                if name in self.feature_names:
                    features.append(feature)
            else:
                # feature_names が未設定の場合のフォールバック
                if "pool" in name:
                    features.append(feature)
    
        # VGG の最終出力も保持（元実装準拠）
        x = feature
        # features は高次解像度側が先になるよう反転（元実装準拠）
        features = features[::-1]
    
        # ====== Room Boundary Prediction (RBP) パス ======
        featuresrbp = []
        for i in range(len(self.rbpups)):
            # upsample + skip
            x = self.rbpups[i](x) + self.rbpcv1[i](features[i + 1])
            x = self.rbpcv2[i](x)
            featuresrbp.append(x)
    
        # 512x512 にアップサンプリング（元実装は K.backend.resize_images を使用）
        logits_cw = tf.keras.backend.resize_images(self.rbpfinal(x), 2, 2, "channels_last")
    
        # ====== Room Types Prediction (RTP) パス ======
        # 元実装ではここで x を VGG の最終 feature に戻す
        x = feature
        for i in range(len(self.rtpups)):
            x = self.rtpups[i](x) + self.rtpcv1[i](features[i + 1])
            x = self.rtpcv2[i](x)
            # RBP の文脈を注入（Non-local Context）
            x = self.non_local_context(featuresrbp[i], x, i)
    
        logits_r = tf.keras.backend.resize_images(self.rtpfinal(x), 2, 2, "channels_last")
    
        # 戻り順は元実装どおり (logits_r, logits_cw)
        return logits_r, logits_cw
