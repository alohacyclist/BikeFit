export type QuantizationLevel = 'fp32' | 'fp16' | 'int8';

export const TFLITE_MODEL_URLS: Record<QuantizationLevel, string> = {
  fp32: 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float32/4',
  fp16: 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4',
  int8: 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4',
};

export const WARMUP_FRAMES = 30;
export const MODEL_INPUT_SIZE = 192;

// Keypoint-Indizes MoveNet (identisch zu bisherigem Stack)
export const KP = {
  LEFT_HIP: 11, RIGHT_HIP: 12,
  LEFT_KNEE: 13, RIGHT_KNEE: 14,
  LEFT_ANKLE: 15, RIGHT_ANKLE: 16,
} as const;

// MoveNet liefert 17 Keypoints in fester Reihenfolge
export const KEYPOINT_NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
] as const;
