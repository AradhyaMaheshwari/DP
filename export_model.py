from ultralytics import YOLO

# Load your trained YOLO11 segmentation model
model = YOLO(r"D:\DP\runs\crack_yolo11s_seg_20261031_111244\weights\best.pt")

# Export to NCNN
model.export(
    format="ncnn",
    imgsz=640
)

print("Model successfully exported to NCNN!")