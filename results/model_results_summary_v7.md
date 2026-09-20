# Final Model Results Summary (Version 7)

Model: YOLOv8m-seg
Dataset: 3,554 annotated images across 7 San Diego neighborhoods
Training: 100 max epochs, early stopping at patience 20

## Overall Metrics
- mAP50: 61.9%
- mAP50-95: 46.2%
- Precision: 58.0%
- Recall: 65.0%

## Per-Class mAP50
| Class         | mAP50  |
|---------------|--------|
| Severe Crack  | 80.9%  |
| Side_walk     | 77.4%  |
| Light Crack   | 77.2%  |
| Bike Lane     | 53.6%  |
| Pothole       | 20.7%  |

## Notes
- Pothole is the weakest performing class. This was tested against a rework
  of 1,711 images (adding more pothole labels), which actually lowered
  Pothole mAP50 to 14.7%, so we reverted to Version 7 as the official result.
- Two model architectures were compared on the same dataset: YOLOv8m-seg
  (61.9% overall) outperformed RF-DETR-seg-medium on this dataset in an
  earlier round, and YOLOv8 was kept as the primary model since it matches
  what was stated in the submitted abstract.
- The raw model weight file (best.pt) is not in GitHub since it is a large
  binary file. The training script (train_v8_colab.py style, same used for
  v7) is in the repo and can regenerate it if needed.
