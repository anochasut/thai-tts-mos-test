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

AB_LABELS = {
    1: "1 - Baseline is much better / Baseline ดีกว่ามาก",
    2: "2 - Baseline is somewhat better / Baseline ดีกว่าเล็กน้อย",
    3: "3 - About the same / ใกล้เคียงกัน ไม่ต่างกัน",
    4: "4 - Candidate is somewhat better / Candidate ดีกว่าเล็กน้อย",
    5: "5 - Candidate is much better / Candidate ดีกว่ามาก",
}

st.set_page_config(page_title="Thai TTS AB Listening Test", page_icon="\U0001F3A7")


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
    pair_idx = manifest.index[manifest["trial_type"] == "pair"].tolist()
    solo_idx = manifest.index[manifest["trial_type"] == "solo"].tolist()
    random.shuffle(pair_idx)
    random.shuffle(solo_idx)
    st.session_state.trial_order = pair_idx + solo_idx  # all "pair" trials first, "solo" always last
    st.session_state.n_pair = len(pair_idx)
    st.session_state.current_idx = 0
    st.session_state.responses = []
    st.session_state.pending_uploads = []
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.stage = "demo"


def start_page():
    manifest = load_manifest()
    n_pair = int((manifest["trial_type"] == "pair").sum())
    n_solo = int((manifest["trial_type"] == "solo").sum())
    st.title("Thai TTS AB Listening Test / แบบทดสอบเปรียบเทียบเสียงสังเคราะห์ภาษาไทย")
    st.markdown(
        f"""
### English
Thank you for taking part in this listening test. It has **two parts**:

**Part 1 -- Comparison ({n_pair} trials):** for each trial you'll hear two clips of
the same sentence, labeled **Baseline** and **Candidate**. Please:

1. **Listen to both** (headphones recommended, in a quiet place).
2. **Compare them**: which sounds better overall (naturalness + how easy it is to
   understand), on a 1-5 scale.
3. **Rate the Candidate's naturalness** on its own, on a 1-5 scale.
4. **Type exactly what you heard** in the Candidate clip, in Thai, to the best of
   your ability -- even if it sounds unnatural or you're not fully sure, write your
   best guess rather than leaving it blank.

**Part 2 -- Baseline only ({n_solo} trials, at the end):** you'll then rate the
naturalness of each Baseline clip on its own, and transcribe it too.

Right after this page, you'll see one **example trial** to help you get familiar
with the task before the real ratings begin.

---

### ภาษาไทย
ขอบคุณที่เข้าร่วมการทดสอบการฟังในครั้งนี้ การทดสอบแบ่งเป็น **2 ส่วน**

**ส่วนที่ 1 -- การเปรียบเทียบ ({n_pair} รายการ):** ในแต่ละรายการ คุณจะได้ฟังเสียง 2 คลิป
ของประโยคเดียวกัน ระบุว่า **Baseline** และ **Candidate** กรุณา:

1. **ฟังทั้งสองคลิป** (แนะนำให้ใช้หูฟัง และอยู่ในที่เงียบ)
2. **เปรียบเทียบ**: เสียงไหนฟังดีกว่าโดยรวม (ทั้งความเป็นธรรมชาติและความเข้าใจง่าย) ในระดับ 1-5
3. **ให้คะแนนความเป็นธรรมชาติของ Candidate** เพียงอย่างเดียว ในระดับ 1-5
4. **พิมพ์สิ่งที่คุณได้ยินในคลิป Candidate ให้ตรงที่สุดเท่าที่จะทำได้** เป็นภาษาไทย แม้ว่าเสียงจะฟังดู
   ไม่เป็นธรรมชาติหรือคุณไม่แน่ใจทั้งหมด กรุณาพิมพ์คำตอบที่ใกล้เคียงที่สุดแทนการเว้นว่างไว้

**ส่วนที่ 2 -- Baseline เพียงอย่างเดียว ({n_solo} รายการ อยู่ท้ายสุด):** คุณจะให้คะแนน
ความเป็นธรรมชาติของคลิป Baseline แต่ละคลิป และถอดเสียงด้วยเช่นกัน

หลังจากหน้านี้ คุณจะได้ฟัง **ตัวอย่างการทดสอบ** หนึ่งรายการ เพื่อให้คุ้นเคยกับลักษณะงาน
ก่อนเริ่มการให้คะแนนจริง
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
    st.title("Example Trial / ตัวอย่างการทดสอบ")

    if demo is None:
        st.markdown(
            "(No demo clip is configured -- you can go straight to the real test.) / "
            "(ไม่มีคลิปตัวอย่างในระบบ -- คุณสามารถเริ่มการทดสอบจริงได้เลย)"
        )
    else:
        st.markdown(
            """
### English
Before the real test, here is one example comparison trial so you know what the
audio quality and the task feel like. This example is **not scored** -- it's just
for you to get familiar with things.

### ภาษาไทย
ก่อนเริ่มการทดสอบจริง นี่คือตัวอย่างรายการเปรียบเทียบหนึ่งรายการ เพื่อให้คุณคุ้นเคยกับ
คุณภาพเสียงและลักษณะของงาน ตัวอย่างนี้ **ไม่มีการให้คะแนน** เป็นเพียงตัวอย่างเพื่อให้คุณคุ้นเคยเท่านั้น
            """
        )
        st.markdown("**Baseline:**")
        with open(os.path.join(DEMO_AUDIO_DIR, demo["baseline_audio_filename"]), "rb") as f:
            st.audio(f.read(), format="audio/wav")
        st.markdown("**Candidate:**")
        with open(os.path.join(DEMO_AUDIO_DIR, demo["candidate_audio_filename"]), "rb") as f:
            st.audio(f.read(), format="audio/wav")

        st.markdown(
            "**How this works:** for each real comparison trial, you'll answer three "
            "questions like the ones below (not scored here, just for practice) -- an "
            "AB comparison, a naturalness rating for the Candidate, and a transcription "
            "of the Candidate. For example, a careful, exact transcription of this clip "
            f"would be:\n\n> {demo['reference_text']}\n\n"
            "**วิธีการทำงาน:** สำหรับแต่ละรายการเปรียบเทียบจริง คุณจะตอบคำถาม 3 ข้อแบบด้านล่างนี้ "
            "(ตัวอย่างนี้ไม่มีการให้คะแนน เป็นเพียงการฝึกฝน) ได้แก่ การเปรียบเทียบ AB การให้คะแนน"
            "ความเป็นธรรมชาติของ Candidate และการถอดเสียง Candidate ตัวอย่างเช่น คำถอดเสียงที่ถูกต้อง"
            f"ของคลิปนี้คือ:\n\n> {demo['reference_text']}"
        )
        st.markdown("**AB comparison scale / ระดับการเปรียบเทียบ AB:**")
        for label in AB_LABELS.values():
            st.markdown(f"- {label}")
        st.markdown("**Naturalness scale / ระดับความเป็นธรรมชาติ:**")
        for label in MOS_LABELS.values():
            st.markdown(f"- {label}")

    if st.button("Start the real test / เริ่มทดสอบจริง", type="primary"):
        st.session_state.stage = "trial"
        st.rerun()


def pair_trial_page(row, idx, phase_idx, phase_total):
    st.progress(
        phase_idx / phase_total,
        text=f"Comparison {phase_idx + 1} of {phase_total} / การเปรียบเทียบที่ {phase_idx + 1} จาก {phase_total}",
    )
    st.markdown(
        "Listen to both clips, then answer all three questions below. / "
        "ฟังเสียงทั้งสองคลิป แล้วตอบคำถามทั้งสามข้อด้านล่าง"
    )

    st.markdown("**Baseline:**")
    with open(os.path.join(AUDIO_DIR, row["baseline_audio_filename"]), "rb") as f:
        st.audio(f.read(), format="audio/wav")
    st.markdown("**Candidate:**")
    with open(os.path.join(AUDIO_DIR, row["candidate_audio_filename"]), "rb") as f:
        st.audio(f.read(), format="audio/wav")

    with st.form(key=f"trial_form_{idx}"):
        st.markdown(
            "**1) Compared to Baseline, how does Candidate sound overall** "
            "(naturalness + how easy it is to understand)? / "
            "**เมื่อเทียบกับ Baseline เสียง Candidate ฟังโดยรวมเป็นอย่างไร** "
            "(ทั้งความเป็นธรรมชาติและความเข้าใจง่าย)?"
        )
        ab_score = st.radio(
            "AB comparison / การเปรียบเทียบ AB",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: AB_LABELS[x],
            index=None,
            horizontal=False,
            label_visibility="collapsed",
        )
        st.markdown(
            "**2) Rate the Candidate's naturalness on its own** / "
            "**ให้คะแนนความเป็นธรรมชาติของ Candidate เพียงอย่างเดียว**"
        )
        candidate_mos = st.radio(
            "Candidate naturalness / ความเป็นธรรมชาติของ Candidate",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: MOS_LABELS[x],
            index=None,
            horizontal=False,
            label_visibility="collapsed",
        )
        transcription = st.text_area(
            "3) Type exactly what you heard in the Candidate clip (in Thai): / "
            "3) พิมพ์สิ่งที่คุณได้ยินในคลิป Candidate ให้ตรงที่สุด (เป็นภาษาไทย):",
            height=80,
        )
        submitted = st.form_submit_button("Submit & continue / ส่งคำตอบและไปต่อ", type="primary")

    if not submitted:
        return None

    if ab_score is None:
        st.error("Please answer the AB comparison before continuing. / กรุณาตอบคำถามเปรียบเทียบ AB ก่อนไปต่อ")
        return None
    if candidate_mos is None:
        st.error(
            "Please rate the Candidate's naturalness before continuing. / "
            "กรุณาให้คะแนนความเป็นธรรมชาติของ Candidate ก่อนไปต่อ"
        )
        return None
    if not transcription.strip():
        st.error(
            "Please type what you heard before continuing (your best guess is fine). / "
            "กรุณาพิมพ์สิ่งที่คุณได้ยินก่อนไปต่อ (เดาที่ใกล้เคียงที่สุดก็ได้)"
        )
        return None

    return {
        "trial_type": "pair",
        "utt_id": row["utt_id"],
        "token": row["token"],
        "ab_score": ab_score,
        "mos": candidate_mos,
        "transcription": transcription.strip(),
    }


def solo_trial_page(row, idx, phase_idx, phase_total):
    st.progress(
        phase_idx / phase_total,
        text=f"Baseline {phase_idx + 1} of {phase_total} / Baseline ที่ {phase_idx + 1} จาก {phase_total}",
    )
    st.markdown(
        "This is the last part: listen to this Baseline clip on its own, then rate it "
        "and transcribe it. / "
        "นี่คือส่วนสุดท้าย: ฟังคลิป Baseline นี้เพียงลำพัง แล้วให้คะแนนและถอดเสียง"
    )

    with open(os.path.join(AUDIO_DIR, row["baseline_audio_filename"]), "rb") as f:
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

    if not submitted:
        return None

    if mos is None:
        st.error("Please select a rating before continuing. / กรุณาเลือกคะแนนก่อนไปต่อ")
        return None
    if not transcription.strip():
        st.error(
            "Please type what you heard before continuing (your best guess is fine). / "
            "กรุณาพิมพ์สิ่งที่คุณได้ยินก่อนไปต่อ (เดาที่ใกล้เคียงที่สุดก็ได้)"
        )
        return None

    return {
        "trial_type": "solo",
        "utt_id": row["utt_id"],
        "token": row["token"],
        "ab_score": None,
        "mos": mos,
        "transcription": transcription.strip(),
    }


def trial_page():
    manifest = load_manifest()
    order = st.session_state.trial_order
    idx = st.session_state.current_idx
    total = len(order)
    n_pair = st.session_state.n_pair
    row = manifest.iloc[order[idx]]

    if idx < n_pair:
        result = pair_trial_page(row, idx, phase_idx=idx, phase_total=n_pair)
    else:
        result = solo_trial_page(row, idx, phase_idx=idx - n_pair, phase_total=total - n_pair)

    if result is None:
        return

    record = {
        "session_id": st.session_state.session_id,
        "listener_id": st.session_state.listener_id,
        "trial_index": idx,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
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
        f"You completed all **{len(st.session_state.responses)}** trials (both parts). "
        "Your responses have been recorded.\n\n"
        f"คุณทำแบบทดสอบครบทั้งหมด **{len(st.session_state.responses)}** รายการ (ทั้งสองส่วน) "
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
