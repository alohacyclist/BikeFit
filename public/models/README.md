# MoveNet TFLite Modelle

Lege hier drei Dateien ab:

```
public/models/
├── movenet-lightning-fp32.tflite
├── movenet-lightning-fp16.tflite
└── movenet-lightning-int8.tflite
```

## Download von Kaggle (login erforderlich)

- **FP32**: https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning
- **FP16**: https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-float16
- **INT8**: https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8

Auf Kaggle: Tab „Model Files" → `.tflite`-Datei herunterladen → in diesen Ordner kopieren und entsprechend umbenennen.

Pfade werden in `src/types/quantization.ts` (`TFLITE_MODEL_URLS`) referenziert.
