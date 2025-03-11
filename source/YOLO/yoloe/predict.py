from ultralytics import YOLOE

model = YOLOE("pretrain/yoloe-v8l-seg.pt")

names = ["person"]
model.set_classes(names, model.get_text_pe(names))

model.predict('ultralytics/assets/bus.jpg', save=True)
