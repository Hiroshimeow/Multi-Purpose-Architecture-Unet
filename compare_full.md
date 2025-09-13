# Training Results - Extended Run 8 (Small Model + Transposed Conv)

## Best mIoU from Training
- 0.2548

## Performance
- fps: 101.40
- latency_ms: 157.79

## Per Class Metrics (Final Eval)
- Road
  - iou: 0.6985
  - dice: 0.8225
- RoadLine
  - iou: 0.0060
  - dice: 0.0120
- Vegetation
  - iou: 0.1769
  - dice: 0.3007
- Sky
  - iou: 0.1739
  - dice: 0.2963
- NoDrivable
  - iou: 0.2169
  - dice: 0.3564

## Classification Report (Final Eval)
- Road
  - precision: 0.9724
  - recall: 0.7127
  - f1-score: 0.8225
  - support: 4560674.0
- RoadLine
  - precision: 0.0095
  - recall: 0.0162
  - f1-score: 0.0120
  - support: 99488.0
- Vegetation
  - precision: 0.2688
  - recall: 0.3411
  - f1-score: 0.3007
  - support: 755281.0
- Sky
  - precision: 0.8240
  - recall: 0.1807
  - f1-score: 0.2963
  - support: 1257851.0
- NoDrivable
  - precision: 0.2351
  - recall: 0.7368
  - f1-score: 0.3564
  - support: 903282.0

### Overall
- accuracy: 0.5810
- macro avg
  - precision: 0.4620
  - recall: 0.3975
  - f1-score: 0.3576
  - support: 7576576.0
- weighted avg
  - precision: 0.7771
  - recall: 0.5810
  - f1-score: 0.6169
  - support: 7576576.0

# Training Results - Extended Run 6 (Large Model + Augmentation)

## Best mIoU from Training
- 0.6079

## Performance
- fps: 43.95
- latency_ms: 364.08

## Per Class Metrics (Final Eval)
- Road
  - iou: 0.8180
  - dice: 0.8999
- RoadLine
  - iou: 0.3905
  - dice: 0.5617
- Vegetation
  - iou: 0.6125
  - dice: 0.7597
- Sky
  - iou: 0.6097
  - dice: 0.7575
- NoDrivable
  - iou: 0.6091
  - dice: 0.7571

## Classification Report (Final Eval)
- Road
  - precision: 0.8836
  - recall: 0.9168
  - f1-score: 0.8999
  - support: 3221724.0
- RoadLine
  - precision: 0.8647
  - recall: 0.4159
  - f1-score: 0.5617
  - support: 351048.0
- Vegetation
  - precision: 0.9175
  - recall: 0.6482
  - f1-score: 0.7597
  - support: 1356365.0
- Sky
  - precision: 0.9829
  - recall: 0.6162
  - f1-score: 0.7575
  - support: 439910.0
- NoDrivable
  - precision: 0.6737
  - recall: 0.8640
  - f1-score: 0.7571
  - support: 2207529.0

### Overall
- accuracy: 0.8127
- macro avg
  - precision: 0.8645
  - recall: 0.6922
  - f1-score: 0.7472
  - support: 7576576.0
- weighted avg
  - precision: 0.8334
  - recall: 0.8127
  - f1-score: 0.8092
  - support: 7576576.0