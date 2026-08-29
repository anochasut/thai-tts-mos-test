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
DEMO_AUDIO_DIR = os.path.join(APP_DIR, "demo_audio")
DEMO_META_PATH = os.path.join(APP_DIR, "demo_meta.json")

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
    1: "1 - Bad (unintelligible) / แย่ (ฟังไม่เข้าใจ)",
    2: "2 - Poor (understood with great difficulty) / ค่อนข้างแย่ (เข้าใจได้ยากมาก)",
    3: "3 - Fair (understood with moderate effort) / ปานกลาง (เข้าใจได้แต่ต้องใช้ความพยายามพอสมควร)",
    4: "4 - Good (understood with little effort) / ดี (เข้าใจได้ง่าย)",
    5: "5 - Excellent (perfectly natural and clear) / ดีเยี่ยม (เป็นธรรมชาติและชัดเจนสมบูรณ์)",
}

st.set_page_config(page_title="Thai TTS Listening Test", page_icon="\U0001F3A7")


@st.cache_data
def load_manifest():
    return pd.read_csv(MANIFEST_PATH, dtype=str)


@st.cache_data
def load_demo_meta():
    if not os.path.exists(DEMO_META_PATH):
        return None
    with open(DEMO_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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
    st.session_state.stage = "demo"


def start_page():
    total = len(load_manifest())
    st.title("Thai TTS Listening Test / แบบทดสอบการฟังเสียงสังเคราะห์ภาษาไทย")
    st.markdown(
        f"""
### English
Thank you for taking part in this listening test.

**What you'll do:** you will hear a series of short Thai audio clips, one at a time.
For each clip, please:

1. **Listen carefully** (headphones recommended, in a quiet place).
2. **Rate the naturalness / intelligibility** of the speech on a 1-5 scale.
3. **Type exactly what you heard**, in Thai, to the best of your ability -- even if it
   sounds unnatural or you are not fully sure, please write your best guess rather than
   leaving it blank.

There are **{total} short clips** in total. Right after this page, you'll see one
**example clip** to help you get familiar with the task before the real ratings begin.

---

### ภาษาไทย
ขอบคุณที่เข้าร่วมการทดสอบการฟังในครั้งนี้

**สิ่งที่คุณจะทำ:** คุณจะได้ฟังคลิปเสียงภาษาไทยสั้น ๆ ทีละคลิป สำหรับแต่ละคลิป กรุณา:

1. **ฟังอย่างตั้งใจ** (แนะนำให้ใช้หูฟัง และอยู่ในที่เงียบ)
2. **ให้คะแนนความเป็นธรรมชาติ/ความเข้าใจง่าย** ของเสียงพูด ในระดับ 1-5
3. **พิมพ์สิ่งที่คุณได้ยินให้ตรงที่สุดเท่าที่จะทำได้** เป็นภาษาไทย แม้ว่าเสียงจะฟังดูไม่เป็นธรรมชาติ
   หรือคุณไม่แน่ใจทั้งหมด กรุณาพิมพ์คำตอบที่ใกล้เคียงที่สุดแทนการเว้นว่างไว้

มีคลิปเสียงทั้งหมด **{total} คลิป** หลังจากหน้านี้ คุณจะได้ฟัง **คลิปตัวอย่าง** หนึ่งคลิป
เพื่อให้คุ้นเคยกับลักษณะงานก่อนเริ่มการให้คะแนนจริง
        """
    )
    listener_id = st.text_input(
        "Your name or listener ID (used only to organize results): / "
        "ชื่อหรือรหัสผู้ฟังของคุณ (ใช้เพื่อจัดระเบียบผลลัพธ์เท่านั้น):"
    )
    if st.button("Continue / ดำเนินการต่อ", type="primary", disabled=not listener_id.strip()):
        st.session_state.listener_id = listener_id.strip()
        init_session()
        st.rerun()


def demo_page():
    demo = load_demo_meta()
    st.title("Example Clip / คลิปตัวอย่าง")

    if demo is None:
        st.markdown(
            "(No demo clip is configured -- you can go straight to the real test.) / "
            "(ไม่มีคลิปตัวอย่างในระบบ -- คุณสามารถเริ่มการทดสอบจริงได้เลย)"
        )
    else:
        st.markdown(
            """
### English
Before the real test, listen to this one example clip so you know what the audio
quality and the task feel like. This example is **not scored** -- it's just for you
to get familiar with things.

### ภาษาไทย
ก่อนเริ่มการทดสอบจริง กรุณาฟังคลิปตัวอย่างนี้ก่อน เพื่อให้คุ้นเคยกับคุณภาพเสียงและลักษณะของงาน
คลิปตัวอย่างนี้ **ไม่มีการให้คะแนน** เป็นเพียงตัวอย่างเพื่อให้คุณคุ้นเคยเท่านั้น
            """
        )
        demo_audio_path = os.path.join(DEMO_AUDIO_DIR, demo["audio_filename"])
        with open(demo_audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

        st.markdown(
            "**How this task works:** for each real clip, you'll rate it 1-5 for "
            "naturalness/intelligibility, and type what you heard. For example, a careful, "
            f"exact transcription of this clip would be:\n\n> {demo['reference_text']}\n\n"
            "**วิธีการทำงาน:** สำหรับแต่ละคลิปจริง คุณจะให้คะแนน 1-5 ด้านความเป็นธรรมชาติ/ความเข้าใจง่าย "
            "และพิมพ์สิ่งที่คุณได้ยิน ตัวอย่างเช่น คำถอดเสียงที่ถูกต้องของคลิปนี้คือ:\n\n"
            f"> {demo['reference_text']}"
        )
        st.markdown("**Rating scale / ระดับคะแนน:**")
        for label in MOS_LABELS.values():
            st.markdown(f"- {label}")

    if st.button("Start the real test / เริ่มทดสอบจริง", type="primary"):
        st.session_state.stage = "trial"
        st.rerun()


def trial_page():
    manifest = load_manifest()
    order = st.session_state.trial_order
    idx = st.session_state.current_idx
    total = len(order)
    row = manifest.iloc[order[idx]]

    st.progress(idx / total, text=f"Clip {idx + 1} of {total} / คลิปที่ {idx + 1} จาก {total}")

    audio_path = os.path.join(AUDIO_DIR, row["audio_filename"])
    with open(audio_path, "rb") as f:
        st.audio(f.read(), format="audio/wav")

    with st.form(key=f"trial_form_{idx}"):
        mos = st.radio(
            "How natural / intelligible was this clip? / "
            "คลิปนี้ฟังเป็นธรรมชาติ/เข้าใจง่ายแค่ไหน?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: MOS_LABELS[x],
            index=None,
            horizontal=False,
        )
        transcription = st.text_area(
            "Type exactly what you heard (in Thai): / "
            "พิมพ์สิ่งที่คุณได้ยินให้ตรงที่สุด (เป็นภาษาไทย):",
            height=80,
        )
        submitted = st.form_submit_button("Submit & continue / ส่งคำตอบและไปต่อ", type="primary")

    if submitted:
        if mos is None:
            st.error(
                "Please select a rating before continuing. / "
                "กรุณาเลือกคะแนนก่อนไปต่อ"
            )
            return
        if not transcription.strip():
            st.error(
                "Please type what you heard before continuing (your best guess is fine). / "
                "กรุณาพิมพ์สิ่งที่คุณได้ยินก่อนไปต่อ (เดาที่ใกล้เคียงที่สุดก็ได้)"
            )
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
            st.warning(
                f"Saved locally, but upload failed ({err}). Will retry automatically. / "
                "บันทึกไว้ในเครื่องแล้ว แต่อัปโหลดไม่สำเร็จ ระบบจะลองใหม่ให้อัตโนมัติ"
            )

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
    st.title("Thank you! \U0001F389 / ขอบคุณที่เข้าร่วม! \U0001F389")
    st.markdown(
        f"You completed all **{len(st.session_state.responses)}** clips. "
        "Your responses have been recorded.\n\n"
        f"คุณทำแบบทดสอบครบทั้งหมด **{len(st.session_state.responses)}** คลิปแล้ว "
        "คำตอบของคุณถูกบันทึกเรียบร้อยแล้ว"
    )
    if st.session_state.pending_uploads:
        st.warning(
            f"{len(st.session_state.pending_uploads)} response(s) could not be uploaded "
            "automatically. Please use the download button below and send the file to the "
            "test organizer as a backup.\n\n"
            f"มีคำตอบ {len(st.session_state.pending_uploads)} รายการที่ไม่สามารถอัปโหลดโดยอัตโนมัติได้ "
            "กรุณากดปุ่มดาวน์โหลดด้านล่างแล้วส่งไฟล์ให้ผู้จัดการทดสอบเป็นข้อมูลสำรอง"
        )
    df = pd.DataFrame(st.session_state.responses)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download my responses (CSV backup) / ดาวน์โหลดคำตอบของฉัน (ไฟล์สำรอง CSV)",
        data=csv_bytes,
        file_name=f"responses_{st.session_state.session_id}.csv",
        mime="text/csv",
    )


def main():
    stage = st.session_state.get("stage", "start")

    if stage == "start":
        start_page()
    elif stage == "demo":
        demo_page()
    else:
        total = len(st.session_state.trial_order)
        if st.session_state.current_idx >= total:
            finish_page()
        else:
            trial_page()


if __name__ == "__main__":
    main()
