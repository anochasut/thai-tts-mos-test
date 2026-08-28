# Thai TTS MOS Listening Test

A blind listening test: for each clip, the listener rates naturalness/intelligibility
(1-5, ITU-T P.800 style) and transcribes what they heard. System identity is hidden
behind opaque tokens (see `manifest.csv`); the real mapping is kept outside this repo.

Each submitted trial is uploaded as a small JSON file to a private Hugging Face
**dataset** repo (configured via `RESULTS_REPO_ID` and `HF_TOKEN` in this app's
Streamlit Community Cloud secrets), so results persist independently of this app.

Deployed on Streamlit Community Cloud. See `../SETUP.md` (in the parent project
folder) for deployment steps and `../analyze_results.py` for scoring (MOS + WER/CER
against ground truth).
