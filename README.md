Voice-Based Emotion Classification

Classifies 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech. I compared classical MFCC models, a CNN trained from scratch, and transfer learning with wav2vec2, and evaluated everything on speakers the model has never heard.

Headline result: wav2vec2 (layer 4) features + logistic regression reach 73.3% ± 6.8% accuracy and 0.726 macro F1 in 4-fold speaker-independent cross-validation (chance = 12.5%).

Kaggle notebook: [add link]
Dataset: RAVDESS on Kaggle
Results
Model	Accuracy	Macro F1	Evaluation
MFCC + Random Forest	0.406	0.360	single split
MFCC + SVM (RBF)	0.456	0.449	single split
CNN on mel-spectrograms (v1)	0.347	0.241	single split
CNN + augmentation (v2)	0.450	0.366	single split
wav2vec2 layer 4 + logistic regression	0.711	0.703	single split
wav2vec2 layer 4 + logistic regression	0.733 ± 0.068	0.726 ± 0.069	4-fold grouped CV

"Single split" means training on actors 1-18 and testing on actors 19-24. No voice appears in both sets.

Cross-validation by fold
Fold	Test actors	Accuracy	Macro F1
1	4, 8, 12, 16, 20, 24	0.803	0.793
2	3, 7, 11, 15, 19, 23	0.689	0.677
3	2, 6, 10, 14, 18, 22	0.794	0.794
4	1, 5, 9, 13, 17, 21	0.644	0.641

The folds happened to be gender-separated (RAVDESS uses even actor IDs for female and odd for male actors): folds 1 and 3 test only female actors, folds 2 and 4 only male actors. Female-test folds scored higher (about 0.80 vs 0.67 accuracy). With only 4 folds this is suggestive, not conclusive, and could reflect differences in how expressive individual actors are rather than anything about the model.

Confusion matrix (test actors 19-24)

Confusion matrix

Largest confusions: fearful → sad (16 clips), sad → calm (11), happy → surprised (8), angry → surprised (7), and calm ↔ neutral. Errors mostly stay within the same arousal level (low: calm, neutral, sad; high: angry, happy, surprised), so the model captures how intense the speech is better than whether it is positive or negative.

Key findings
Duplicate files. The copy of the dataset mounted in my notebook contained every file twice (2,880 paths, 1,440 unique filenames). I verified with MD5 hashes that the copies are byte-identical and removed them. Keeping them would put a clip in training and its twin in the test set (data leakage).
Speaker-independent evaluation. All splits are by actor, so the model cannot score well by recognising a voice.
Transfer learning wins on small data. With only 1,080 training clips, a CNN from scratch did not beat an SVM on MFCCs, while frozen wav2vec2 features gave a large gain.
Middle layers work best. Layer 4 reached 0.653 validation macro F1, while the final layer reached only 0.501. Layers were compared on a validation split (train actors 1-14, validation actors 15-18), not on the test actors.
Hand-crafted prosody features added nothing. Adding 8 prosody features (energy variation, spectral centroid, zero-crossing rate, pitch level, variation and range) moved macro F1 from 0.726 to 0.729, which is within noise, and angry recall stayed at 0.807. The final model uses wav2vec2 features only.
Method
Data: RAVDESS speech, 24 professional actors, 1,440 clips. The emotion label and actor ID are decoded from each filename.
MFCC baselines: mean and standard deviation of 40 MFCCs per clip (80 features), with scaling, SVM and Random Forest.
CNN: 64-band mel-spectrograms fed into a small 3-block CNN. Version 2 adds silence trimming, per-clip loudness normalisation, SpecAugment-style time and frequency masking, time shifting, and cosine learning-rate decay.
wav2vec2: the pretrained facebook/wav2vec2-base is used as a frozen feature extractor. Hidden states of each layer are averaged over time, and a logistic regression is trained on the chosen layer.
Evaluation: accuracy, macro F1, per-class reports, confusion matrix, and 4-fold cross-validation grouped by actor.
Demo

A Gradio app takes an uploaded or recorded clip and returns probabilities for the 8 emotions.

bash
pip install -r requirements.txt
python app.py

app.py needs emotion_model.joblib (scaler + classifier, trained on all 24 actors) in the same folder. The first run downloads wav2vec2-base (about 380 MB). Record 3-4 seconds of expressive speech for best results.

The demo removes silent gaps (top_db=25) before feature extraction, while training features used end-trimming only (top_db=30). I did not re-evaluate this small preprocessing difference. The reported accuracy comes from the cross-validation above, not from the demo model.

Limitations
RAVDESS is acted, studio-quality, North American English speech with only two sentences. Real-world accuracy will be lower.
Informal tests on my own voice were unreliable. After removing silence, most emotions were predicted correctly, but angry speech was predicted as neutral or happy. This is domain shift (training data differs from real recordings), and these few clips are anecdotal and not part of the reported results.
Layer 4 was chosen using actors 15-18, who also appear in some cross-validation test folds, so the cross-validation estimate is slightly optimistic.
Only 4 folds were used, and results vary noticeably between speaker groups (0.64 to 0.80 accuracy).
Future work
Add CREMA-D and TESS for more speakers and sentences, then re-test on real recordings.
Use a gender-balanced cross-validation split and more folds.
Fine-tune wav2vec2 instead of freezing it.
