"""Speech Emotion Recognition demo.

wav2vec2-base (frozen, layer 4 hidden states, mean-pooled over time)
+ StandardScaler + logistic regression, trained on RAVDESS.

Run:  python app.py
Needs emotion_model.joblib in the same folder (saved from the notebook).
"""
import joblib
import librosa
import numpy as np
import torch
import gradio as gr
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SR = 16000                  # wav2vec2 expects 16 kHz audio
MODEL_NAME = "facebook/wav2vec2-base"
MIN_SECONDS = 0.5           # clips shorter than this are too short to classify

device = "cuda" if torch.cuda.is_available() else "cpu"

bundle = joblib.load("emotion_model.joblib")
scaler, clf, layer = bundle["scaler"], bundle["clf"], bundle["layer"]

extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
w2v = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device).eval()


def remove_silence(y, top_db=25):
    """Drop silent gaps so long pauses don't dilute the averaged features."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    return np.concatenate([y[s:e] for s, e in intervals])


def predict(audio_path):
    if audio_path is None:
        return {}
    y, _ = librosa.load(audio_path, sr=SR)
    y = remove_silence(y)
    if len(y) < MIN_SECONDS * SR:
        raise gr.Error("Audio is too short. Please record at least 1 second of speech.")

    inputs = extractor(y, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        out = w2v(inputs.input_values.to(device), output_hidden_states=True)

    feat = out.hidden_states[layer].mean(dim=1).squeeze(0).cpu().numpy().reshape(1, -1)
    probs = clf.predict_proba(scaler.transform(feat))[0]
    return {str(c): float(p) for c, p in zip(clf.classes_, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload or record speech (3-4 seconds works best)"),
    outputs=gr.Label(num_top_classes=8, label="Predicted emotion"),
    title="Speech Emotion Recognition",
    description=(
        "wav2vec2 (layer 4) + logistic regression, trained on the acted RAVDESS dataset. "
        "Accuracy on unseen speakers is about 73% on RAVDESS; expect lower on real-world audio."
    ),
)

if __name__ == "__main__":
    demo.launch()
