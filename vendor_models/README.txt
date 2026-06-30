把老板给你的模型放到这里：

vendor_models/valorant_320_v11n.onnx
vendor_models/valorant_256_v11n.onnx

config.yaml 默认使用 vendor_models/valorant_320_v11n.onnx。
如果要用 256 模型，把 config.yaml 中的 model.path 和 model.imgsz 分别改成：
model.path: vendor_models/valorant_256_v11n.onnx
model.imgsz: 256
