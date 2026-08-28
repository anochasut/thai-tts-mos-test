import io
import json
import os
import random
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from huggingface_hub import HfApi

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(APP_DIR, "manifest.csv")
AUDIO_DIR = os.path.join(APP_DIR, "audio")

def _get_secret(name: str, default: str = "") -> str:
    # Streamlit Community Cloud injects secrets via st.secrets (TOML you paste into
    # the app's "Secrets" dashboard), not as plain environment variables. Fall back
    # to os.environ so this also works if ever hosted somewhere env-var-based.
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


RESULTS_REPO_ID = _get_secret("RESULTS_REPO_ID")
HF_TOKEN = _get_secret("HF_TOKEN")

MOS_LABELS = {
    1: "1 - Bad (unintelligible)",
    2: "2 - Poor (understood with great difficulty)",
    3: "3 - Fair (understood with moderate effort)",
    4: "4 - Good (understood with little effort)",
    5: "5 - Excellent (perfectly natural and clear)",
}

st.set_page_config(page_title="Thai TTS Listening Test", page_icon="\U0001F3A7")


@st.cache_data
def load_manifest():
    return pd.read_csv(MANIFEST_PATH, dtype=str)


def get_api():
    if not HF_TOKEN or not RESULTS_REPO_ID:
        return None
    return HfApi(token=HF_TOKEN)


def upload_submission(record: dict):
    api = get_api()
    if api is None:
        return False, "not configured"
    payload = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    path_in_repo = f"{record['session_id']}/{record['trial_index']:03d}_{record['utt_id']}.json"
    try:
        api.upload_file(
            path_or_fileobj=io.BytesIO(payload),
            path_in_repo=path_in_repo,
            repo_id=RESULTS_REPO_ID,
            repo_type="dataset",
            commit_message=f"submission {record['session_id']} #{record['trial_index']}",
        )
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def init_session():
    manifest = load_manifest()
    order = list(range(len(manifest)))
    random.shuffle(order)
    st.session_state.trial_order = order
    st.session_state.current_idx = 0
    st.session_state.responses = []
    st.session_state.pending_uploads = []
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.started = True


def start_page():
    st.title("Thai TTS Listening Test")
    st.markdown(
        """
Thank you for taking part in this listening test.

**What you'll do:** you will hear a series of short Thai audio clips, one at a time.
For each clip, please:

1. **Listen carefully** (headphones recommended, in a quiet place).
2. **Rate the naturalness / intelligibility** of the speech on a 1-5 scale.
3. **Type exactly what you heard**, in Thai, to the best of your ability -- even if it
   sounds unnatural or you are not fully sure, please write your best guess rather than
   leaving it blank.

There are **95 short clips** in total. The session takes roughly 30-45 minutes.
You may take breaks between clips -- your progress is saved after every clip.
        """
    )
    listener_id = st.text_input("Your name or listener ID (used only to organize results):")
    if st.button("Begin test", type="primary", disabled=not listener_id.strip()):
        st.session_state.listener_id = listener_id.strip()
        init_session()
        st.rerun()


def trial_page():
    manifest = load_manifest()
    order = st.session_state.trial_order
    idx = st.session_state.current_idx
    total = len(order)
    row = manifest.iloc[order[idx]]

    st.progress(idx / total, text=f"Clip {idx + 1} of {total}")

    audio_path = os.path.join(AUDIO_DIR, row["audio_filename"])
    with open(audio_path, "rb") as f:
        st.audio(f.read(), format="audio/wav")

    with st.form(key=f"trial_form_{idx}"):
        mos = st.radio(
            "How natural / intelligible was this clip?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: MOS_LABELS[x],
            index=None,
            horizontal=False,
        )
        transcription = st.text_area(
            "Type exactly what you heard (in Thai):",
            height=80,
        )
        submitted = st.form_submit_button("Submit & continue", type="primary")

    if submitted:
        if mos is None:
            st.error("Please select a rating before continuing.")
            return
        if not transcription.strip():
            st.error("Please type what you heard before continuing (your best guess is fine).")
            return

        record = {
            "session_id": st.session_state.session_id,
            "listener_id": st.session_state.listener_id,
            "trial_index": idx,
            "utt_id": row["utt_id"],
            "token": row["token"],
            "mos": mos,
            "transcription": transcription.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        st.session_state.responses.append(record)

        ok, err = upload_submission(record)
        if not ok:
            st.session_state.pending_uploads.append(record)
            st.warning(f"Saved locally, but upload failed ({err}). Will retry automatically.")

        st.session_state.current_idx += 1
        st.rerun()


def retry_pending():
    still_pending = []
    for record in st.session_state.pending_uploads:
        ok, _ = upload_submission(record)
        if not ok:
            still_pending.append(record)
    st.session_state.pending_uploads = still_pending


def finish_page():
    retry_pending()
    st.title("Thank you! \U0001F389")
    st.markdown(
        f"You completed all **{len(st.session_state.responses)}** clips. "
        "Your responses have been recorded."
    )
    if st.session_state.pending_uploads:
        st.warning(
            f"{len(st.session_state.pending_uploads)} response(s) could not be uploaded "
            "automatically. Please use the download button below and send the file to the "
            "test organizer as a backup."
        )
    df = pd.DataFrame(st.session_state.responses)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download my responses (CSV backup)",
        data=csv_bytes,
        file_name=f"responses_{st.session_state.session_id}.csv",
        mime="text/csv",
    )


def main():
    if not st.session_state.get("started"):
        start_page()
        return

    total = len(st.session_state.trial_order)
    if st.session_state.current_idx >= total:
        finish_page()
    else:
        trial_page()


if __name__ == "__main__":
    main()
