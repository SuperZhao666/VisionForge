#!/usr/bin/env python3
"""二分类 CNN 训练脚本：敌人 vs 非敌人（新版默认 RGB+mask 四通道输入）。

新版目标：
- 不再只让模型看二值 mask 形状。
- 默认输入为 48×48×4：RGB 原图 ROI + mask 辅助通道。
- 保留旧配置和训练流程；如果检测到旧 1 通道 best_model.keras，会自动从头训练 4 通道模型。
"""
import os, sys, subprocess, shutil, random, importlib

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", MIRROR, pkg])


for pkg, imp in [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pyyaml", "yaml"),
    ("tensorflow", "tensorflow"),
    ("scikit-learn", "sklearn"),
]:
    ensure(pkg, imp)

import numpy as np
import cv2
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import Config
from detector import build_color_mask as build_shared_color_mask
from model_publish import atomic_publish_onnx
from model_input import adapt_batch_layout, infer_input_layout, make_model_input

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 60
TRAIN_CFG = {}
TRAIN_CFG_OBJ = None
FIRE_DIR = "dataset/fire"        # 正样本：敌人
NO_FIRE_DIR = "dataset/no_fire"  # 负样本：非敌人
OUTPUT_ONNX = "fire_model.onnx"
OUTPUT_KERAS = "fire_model.keras"
TEMP_SAVEDMODEL = "temp_saved_model"
BEST_KERAS = "best_model.keras"
SUPPORTED_CHANNELS = {1, 3, 4}


def load_config():
    return Config.from_yaml()


def load_color_thresholds(cfg):
    lower = cfg.color_lower
    upper = cfg.color_upper
    return np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)


TRAIN_CFG_OBJ = load_config()
TRAIN_CFG = TRAIN_CFG_OBJ.__dict__
OUTPUT_ONNX = str(TRAIN_CFG.get("model_path", OUTPUT_ONNX))
IMG_SIZE = max(8, int(TRAIN_CFG.get("img_size", IMG_SIZE)))
MODEL_INPUT_CHANNELS = int(TRAIN_CFG.get("model_input_channels", 4) or 4)
if MODEL_INPUT_CHANNELS not in SUPPORTED_CHANNELS:
    print(f"[警告] model_input_channels={MODEL_INPUT_CHANNELS} 不支持，已改为 4")
    MODEL_INPUT_CHANNELS = 4
BATCH_SIZE = max(1, int(TRAIN_CFG.get("train_batch_size", BATCH_SIZE)))
EPOCHS = max(1, int(TRAIN_CFG.get("train_epochs", EPOCHS)))
COLOR_LOWER, COLOR_UPPER = load_color_thresholds(TRAIN_CFG)
print(f"[配置] 颜色阈值: lower={COLOR_LOWER.tolist()} upper={COLOR_UPPER.tolist()}")
print(f"[配置] 训练参数: epochs={EPOCHS} batch={BATCH_SIZE} augment={bool(TRAIN_CFG.get('augment_training_data', True))}")
print(f"[配置] 模型输入: {IMG_SIZE}×{IMG_SIZE}×{MODEL_INPUT_CHANNELS} ({'RGB+mask' if MODEL_INPUT_CHANNELS == 4 else 'RGB' if MODEL_INPUT_CHANNELS == 3 else 'mask-only'})")


def get_loss():
    gamma = float(TRAIN_CFG.get("focal_loss_gamma", 0.0) or 0.0)
    if gamma > 0 and hasattr(tf.keras.losses, "BinaryFocalCrossentropy"):
        return tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma)
    return "binary_crossentropy"


def build_color_mask(img_bgr):
    """训练样本来自 cv2.imread，通道顺序固定为 BGR。"""
    return build_shared_color_mask(img_bgr, TRAIN_CFG_OBJ, source_color="BGR")


def augment_sample(sample):
    """对 RGB+mask 同步做几何增强；形态学扰动只作用于 mask 通道。"""
    img = sample.copy()
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.4:
        m = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), random.uniform(-10, 10), 1.0)
        img = cv2.warpAffine(img, m, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if random.random() < 0.3:
        tx, ty = random.randint(-3, 3), random.randint(-3, 3)
        m = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, m, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if random.random() < 0.3:
        kernel = np.ones((2, 2), np.uint8)
        if MODEL_INPUT_CHANNELS == 1:
            ch = img[:, :, 0]
            ch = cv2.dilate(ch, kernel, iterations=1) if random.random() < 0.5 else cv2.erode(ch, kernel, iterations=1)
            img[:, :, 0] = ch
        elif MODEL_INPUT_CHANNELS == 4:
            ch = img[:, :, 3]
            ch = cv2.dilate(ch, kernel, iterations=1) if random.random() < 0.5 else cv2.erode(ch, kernel, iterations=1)
            img[:, :, 3] = ch
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def extract_rois_from_dir(dir_path, label):
    """从目录中提取所有紫色轮廓ROI，并构造统一模型输入。"""
    rois = []
    labels = []
    if not os.path.isdir(dir_path):
        return rois, labels

    files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")))
    max_files = int(TRAIN_CFG.get("max_samples_per_class", 0) or 0)
    if max_files > 0:
        files = files[-max_files:]
    print(f"  [{dir_path}] 文件数: {len(files)}")

    for fname in files:
        full_path = os.path.join(dir_path, fname)
        img_bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        mask = build_color_mask(img_bgr)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 5 or h < 5:
                continue
            roi_mask = mask[y:y + h, x:x + w]
            if np.count_nonzero(roi_mask) < 10:
                continue
            roi_bgr = img_bgr[y:y + h, x:x + w]
            try:
                sample = make_model_input(roi_bgr, roi_mask, IMG_SIZE, channels=MODEL_INPUT_CHANNELS, source_color="BGR")
            except Exception:
                continue
            rois.append(sample)
            labels.append(label)

    return rois, labels


def load_data():
    print("[加载] 提取正样本(敌人)...")
    fire_rois, fire_labels = extract_rois_from_dir(FIRE_DIR, 1)
    print(f"  正样本ROI: {len(fire_rois)}")

    print("[加载] 提取负样本(非敌人)...")
    nofire_rois, nofire_labels = extract_rois_from_dir(NO_FIRE_DIR, 0)
    print(f"  负样本ROI: {len(nofire_rois)}")

    if len(fire_rois) < 10:
        print(f"\n[严重错误] 敌人样本不足 10！当前: {len(fire_rois)}")
        print("请先在游戏中对准敌人按 ] 键截取正样本！")
        sys.exit(1)

    if len(nofire_rois) < 10:
        print(f"\n[严重错误] 负样本不足 10！当前: {len(nofire_rois)}")
        print("请先在游戏中按 ~ 键截取非敌人背景！")
        sys.exit(1)

    augment_enabled = bool(TRAIN_CFG.get("augment_training_data", True))
    X, Y = [], []
    for img, label in zip(fire_rois, fire_labels):
        X.append(img)
        Y.append(label)
        if augment_enabled:
            for _ in range(3):
                X.append(augment_sample(img))
                Y.append(label)

    # 负样本增强数量与正样本平衡。
    neg_augment = min(8, max(1, len(fire_rois) * 4 // max(len(nofire_rois), 1))) if augment_enabled else 0
    for img, label in zip(nofire_rois, nofire_labels):
        X.append(img)
        Y.append(label)
        for _ in range(neg_augment):
            X.append(augment_sample(img))
            Y.append(label)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], Y[indices]


def model_input_channels(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) == 4 and shape[-1] in SUPPORTED_CHANNELS:
        return int(shape[-1])
    if len(shape) == 4 and shape[1] in SUPPORTED_CHANNELS:
        return int(shape[1])
    return None


def build_model():
    input_img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, MODEL_INPUT_CHANNELS))
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(input_img, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=get_loss(), metrics=["accuracy"])
    return model


def make_synthetic(kind: str) -> np.ndarray:
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if kind == "enemy":
        cv2.circle(mask, (IMG_SIZE // 2, IMG_SIZE // 5), max(3, IMG_SIZE // 10), 255, 1)
        cv2.rectangle(mask, (IMG_SIZE // 2 - 6, IMG_SIZE // 3), (IMG_SIZE // 2 + 6, IMG_SIZE * 3 // 4), 255, 1)
        cv2.rectangle(mask, (IMG_SIZE // 2 - 15, IMG_SIZE // 3 + 3), (IMG_SIZE // 2 - 6, IMG_SIZE // 2 + 4), 255, 1)
        cv2.rectangle(mask, (IMG_SIZE // 2 + 6, IMG_SIZE // 3 + 3), (IMG_SIZE // 2 + 15, IMG_SIZE // 2 + 4), 255, 1)
    else:
        cv2.circle(mask, (IMG_SIZE // 2, IMG_SIZE // 2), max(8, IMG_SIZE // 3), 255, -1)

    rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    rgb[mask > 0] = (185, 80, 255)  # RGB 紫色，仅用于 sanity check。
    return make_model_input(rgb, mask, IMG_SIZE, channels=MODEL_INPUT_CHANNELS, source_color="RGB")


def main():
    print("=" * 60)
    print("[步骤1] 加载正负样本并提取 RGB+mask ROI...")
    X, Y = load_data()
    print(f"[数据] 总样本(含增强): {len(X)} (敌人={int(Y.sum())} 非敌人={int(len(Y)-Y.sum())})")
    print(f"[数据] X shape={X.shape}")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    print(f"[数据] 训练集: {len(X_train)}, 验证集: {len(X_val)}")

    model = None
    if os.path.exists(BEST_KERAS):
        try:
            loaded = models.load_model(BEST_KERAS, compile=False)
            old_ch = model_input_channels(loaded)
            if old_ch == MODEL_INPUT_CHANNELS:
                print(f"[加载] 检测到兼容模型 {BEST_KERAS} ({old_ch}ch)，继续训练...")
                model = loaded
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                              loss=get_loss(), metrics=["accuracy"])
            else:
                print(f"[警告] {BEST_KERAS} 是 {old_ch}ch，当前要求 {MODEL_INPUT_CHANNELS}ch；将从头构建新模型。")
        except Exception as e:
            print(f"[警告] 旧模型加载失败，将从头训练: {e}")

    if model is None:
        print("[构建] 从头开始构建新模型...")
        model = build_model()
        model.summary()

    print("\n" + "=" * 60)
    print(f"[步骤3] 开始训练 (epochs={EPOCHS}, batch={BATCH_SIZE})...")

    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(BEST_KERAS, monitor="val_loss", save_best_only=True, verbose=1)

    history = model.fit(
        X_train, Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, Y_val),
        verbose=1,
        callbacks=[early_stop, checkpoint],
        initial_epoch=0,
    )

    print(f"\n[结果] 验证集准确率: {history.history['val_accuracy'][-1]:.4f}")
    model.save(OUTPUT_KERAS)

    print("\n" + "=" * 60)
    print("[步骤4] 转换为 ONNX...")
    temp_onnx = OUTPUT_ONNX + ".tmp"
    try:
        if os.path.exists(temp_onnx):
            os.remove(temp_onnx)
        try:
            import tf2onnx  # noqa: F401
        except ImportError:
            print("[安装] 缺少 tf2onnx，正在安装...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", MIRROR, "tf2onnx"])

        shutil.rmtree(TEMP_SAVEDMODEL, ignore_errors=True)
        tf.saved_model.save(model, TEMP_SAVEDMODEL)
        cmd = [sys.executable, "-m", "tf2onnx.convert",
               "--saved-model", TEMP_SAVEDMODEL,
               "--output", temp_onnx, "--opset", "13"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[成功] ONNX 临时文件已保存: {temp_onnx}")

        try:
            import onnx
            from onnxsim import simplify
            m = onnx.load(temp_onnx)
            m_s, ok = simplify(m)
            if ok:
                onnx.save(m_s, temp_onnx)
                print("[成功] ONNX 图简化完成")
        except Exception as e:
            print(f"[警告] ONNX 图简化失败: {e}")

        if bool(TRAIN_CFG.get("quantize_onnx", False)):
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                q_path = temp_onnx.replace(".onnx.tmp", "_int8.onnx.tmp")
                quantize_dynamic(temp_onnx, q_path, weight_type=QuantType.QInt8)
                os.replace(q_path, temp_onnx)
                print("[成功] ONNX 动态量化完成")
            except Exception as e:
                print(f"[警告] ONNX 量化失败，继续使用未量化模型: {e}")

        print("\n[步骤5] 验证模型...")
        import onnxruntime as ort_test
        sess = ort_test.InferenceSession(temp_onnx, providers=["CPUExecutionProvider"])
        inp_meta = sess.get_inputs()[0]
        inp_name = inp_meta.name
        out_name = sess.get_outputs()[0].name
        onnx_ch, onnx_cf = infer_input_layout(inp_meta.shape)
        print(f"  ONNX 输入: shape={inp_meta.shape} channels={onnx_ch} layout={'NCHW' if onnx_cf else 'NHWC'}")

        n_val = min(20, len(X_val))
        val_x = adapt_batch_layout(X_val[:n_val], onnx_cf)
        val_pred = sess.run([out_name], {inp_name: val_x})[0].flatten()
        val_labels = Y_val[:n_val]
        correct = 0
        for i in range(n_val):
            pred_label = 1 if val_pred[i] > 0.5 else 0
            status = "✓" if pred_label == int(val_labels[i]) else "✗"
            correct += (pred_label == int(val_labels[i]))
            print(f"  样本{i}: pred={val_pred[i]:.4f} label={int(val_labels[i])} {status}")
        print(f"\n  验证准确率: {correct}/{n_val} = {correct / max(1, n_val) * 100:.0f}%")

        enemy = make_synthetic("enemy").reshape(1, IMG_SIZE, IMG_SIZE, MODEL_INPUT_CHANNELS)
        circle = make_synthetic("circle").reshape(1, IMG_SIZE, IMG_SIZE, MODEL_INPUT_CHANNELS)
        enemy = adapt_batch_layout(enemy, onnx_cf)
        circle = adapt_batch_layout(circle, onnx_cf)
        ep = float(sess.run([out_name], {inp_name: enemy})[0].reshape(-1)[0])
        cp = float(sess.run([out_name], {inp_name: circle})[0].reshape(-1)[0])
        print(f"\n  合成人形 prob: {ep:.4f} (仅作 sanity check)")
        print(f"  合成圆球 prob: {cp:.4f} (仅作 sanity check)")
        print("[提示] 最终是否达标请以 eval_model.py 的 FP/FPR/Recall 为准。")
        meta = atomic_publish_onnx(temp_onnx, OUTPUT_ONNX, IMG_SIZE)
        print(f"[成功] ONNX 已原子发布: {OUTPUT_ONNX} ready={OUTPUT_ONNX}.ready.json meta={meta}")

    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)
    finally:
        if os.path.exists(TEMP_SAVEDMODEL):
            shutil.rmtree(TEMP_SAVEDMODEL, ignore_errors=True)
        if os.path.exists(temp_onnx):
            try:
                os.remove(temp_onnx)
            except OSError:
                pass

    print("\n" + "=" * 60)
    print("[完成] 二分类模型训练结束")


if __name__ == "__main__":
    main()
