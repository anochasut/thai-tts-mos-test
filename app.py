import hashlib
import io
import json
import os
import random
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(APP_DIR, "manifest.csv")
AUDIO_DIR = os.path.join(APP_DIR, "audio")
DEMO_AUDIO_DIR = os.path.join(APP_DIR, "demo_audio")
DEMO_META_PATH = os.path.join(APP_DIR, "demo_meta.json")

SESSION_FILENAME = "_session.json"


def _get_secret(name: str, default: str = "") -> str:
    # Streamlit Community Cloud injects secrets via st.secrets (TOML you paste into
    # the app's "Secrets" dashboard), not as plain environment variables. Fall back
    # to os.environ so this also works if ever hosted somewhere env-var-based.
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def _get_email_list(name: str) -> set:
    """Comma/whitespace-separated list of emails from secrets -> lowercased set."""
    raw = _get_secret(name, "")
    return {e.strip().lower() for e in re.split(r"[,\s]+", raw) if e.strip()}


RESULTS_REPO_ID = _get_secret("RESULTS_REPO_ID")
HF_TOKEN = _get_secret("HF_TOKEN")
ADMIN_EMAILS = _get_email_list("ADMIN_EMAILS")
ALLOWED_EMAILS = _get_email_list("ALLOWED_EMAILS")  # empty = anyone may participate

MOS_LABELS = {
    1: "1 - Bad (unintelligible) / แย่ (ฟังไม่เข้าใจ)",
    2: "2 - Poor (understood with great difficulty) / ค่อนข้างแย่ (เข้าใจได้ยากมาก)",
    3: "3 - Fair (understood with moderate effort) / ปานกลาง (เข้าใจได้แต่ต้องใช้ความพยายามพอสมควร)",
    4: "4 - Good (understood with little effort) / ดี (เข้าใจได้ง่าย)",
    5: "5 - Excellent (perfectly natural and clear) / ดีเยี่ยม (เป็นธรรมชาติและชัดเจนสมบูรณ์)",
}

st.set_page_config(page_title="Thai TTS Listening Test", page_icon="\U0001F3A7", layout="centered")


# --------------------------------------------------------------------------------------
# Test set
# --------------------------------------------------------------------------------------
@st.cache_data
def load_manifest():
    return pd.read_csv(MANIFEST_PATH, dtype=str)


@st.cache_data
def manifest_id() -> str:
    """
    Short hash of manifest.csv. Results are stored under this id so that each round of
    audio (each prepare_manifest.py run that changes the set) is kept separate -- an
    old round's answers can never be resumed into, or mixed with, a new one.
    """
    with open(MANIFEST_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


@st.cache_data
def load_demo_meta():
    if not os.path.exists(DEMO_META_PATH):
        return None
    with open(DEMO_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Authentication (Streamlit native OIDC -- see SETUP.md for the Google setup)
# --------------------------------------------------------------------------------------
def auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def auth_provider() -> str:
    """'' when secrets use a single flat [auth] block, else the provider sub-section name."""
    try:
        auth = st.secrets["auth"]
        for provider in ("google", "microsoft", "auth0", "okta"):
            if provider in auth:
                return provider
    except Exception:
        pass
    return ""


def current_user():
    """{'email','name'} for the logged-in user, or None."""
    if not auth_configured():
        # Local/dev fallback: no OIDC configured, identify by typed-in email instead.
        return st.session_state.get("dev_user")
    try:
        if not st.user.is_logged_in:
            return None
    except Exception:
        return None
    email = (getattr(st.user, "email", None) or "").strip().lower()
    name = (getattr(st.user, "name", None) or "").strip() or email
    if not email:
        return None
    return {"email": email, "name": name}


def login_gate():
    """Render the login screen. Returns the user dict, or None if not signed in yet."""
    user = current_user()
    if user is not None:
        return user

    st.title("Thai TTS Listening Test / แบบทดสอบการฟังเสียงสังเคราะห์ภาษาไทย")
    st.markdown(
        """
### English
Please sign in with your **Google account** to begin.

Signing in lets you **stop whenever you like and come back later** -- your answers are
saved as you go, and you can also **go back and change** any answer before you finish.

### ภาษาไทย
กรุณาเข้าสู่ระบบด้วย **บัญชี Google** ของคุณเพื่อเริ่มทำแบบทดสอบ

การเข้าสู่ระบบช่วยให้คุณ **หยุดพักเมื่อไรก็ได้แล้วกลับมาทำต่อภายหลัง** คำตอบของคุณจะถูกบันทึกไว้ตลอด
และคุณยัง **ย้อนกลับไปแก้ไขคำตอบ** ใดก็ได้ก่อนที่จะส่งแบบทดสอบ
        """
    )

    if auth_configured():
        if st.button("Sign in with Google / เข้าสู่ระบบด้วย Google", type="primary"):
            provider = auth_provider()
            if provider:
                st.login(provider)
            else:
                st.login()
        st.stop()

    # Dev fallback so the app still runs locally without OAuth secrets configured.
    st.info(
        "Google sign-in is not configured on this deployment -- running in local/dev mode. "
        "Enter an email address to identify this session."
    )
    email = st.text_input("Email")
    if st.button("Continue", type="primary", disabled=not email.strip()):
        st.session_state.dev_user = {
            "email": email.strip().lower(),
            "name": email.strip(),
        }
        st.rerun()
    st.stop()


def do_logout():
    if auth_configured():
        st.logout()
    else:
        st.session_state.clear()
        st.rerun()


def session_id_for(email: str) -> str:
    """Stable, path-safe id per participant. Hash suffix guards against slug collisions."""
    slug = re.sub(r"[^a-z0-9]+", "_", email.lower()).strip("_") or "listener"
    digest = hashlib.sha256(email.lower().encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{digest}"


# --------------------------------------------------------------------------------------
# Remote store (private Hugging Face dataset repo)
#   {manifest_id}/{session_id}/_session.json          <- trial order + participant info
#   {manifest_id}/{session_id}/{idx:03d}_{utt}.json   <- one file per answered clip
# --------------------------------------------------------------------------------------
def get_api():
    if not HF_TOKEN or not RESULTS_REPO_ID:
        return None
    return HfApi(token=HF_TOKEN)


def _upload_json(path_in_repo: str, payload: dict, commit_message: str):
    api = get_api()
    if api is None:
        return False, "results storage is not configured"
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=path_in_repo,
            repo_id=RESULTS_REPO_ID,
            repo_type="dataset",
            commit_message=commit_message,
        )
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _download_json(path_in_repo: str):
    try:
        local = hf_hub_download(
            RESULTS_REPO_ID,
            path_in_repo,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def trial_path(session_id: str, trial_index: int, utt_id: str) -> str:
    return f"{manifest_id()}/{session_id}/{trial_index:03d}_{utt_id}.json"


def session_path(session_id: str) -> str:
    return f"{manifest_id()}/{session_id}/{SESSION_FILENAME}"


@st.cache_data(ttl=30, show_spinner=False)
def list_repo_files_cached():
    api = get_api()
    if api is None:
        return []
    try:
        return api.list_repo_files(RESULTS_REPO_ID, repo_type="dataset")
    except Exception:  # noqa: BLE001
        return []


@st.cache_data(ttl=30, show_spinner=False)
def load_session_meta_cached(path_in_repo: str):
    return _download_json(path_in_repo)


def load_remote_progress(session_id: str):
    """
    Pull this participant's saved state back down: their fixed clip order plus every
    answer they've already given. Returns (session_meta or None, {trial_index: record}).
    """
    if get_api() is None:
        return None, {}

    prefix = f"{manifest_id()}/{session_id}/"
    try:
        # One batched, parallel fetch of this participant's folder -- downloading each
        # answer file separately would mean ~85 round-trips on every resume.
        local_root = snapshot_download(
            RESULTS_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            allow_patterns=[f"{prefix}*"],
        )
    except Exception:  # noqa: BLE001
        return None, {}

    folder = os.path.join(local_root, manifest_id(), session_id)
    if not os.path.isdir(folder):
        return None, {}

    meta = None
    answers = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, name), "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        if name == SESSION_FILENAME:
            meta = payload
        elif "trial_index" in payload:
            answers[int(payload["trial_index"])] = payload
    return meta, answers


# --------------------------------------------------------------------------------------
# Local session state
# --------------------------------------------------------------------------------------
def ensure_session_loaded(user: dict):
    """Restore (or create) this participant's test session. Runs once per browser session."""
    session_id = session_id_for(user["email"])
    if st.session_state.get("loaded_for") == session_id:
        return

    manifest = load_manifest()
    total = len(manifest)

    with st.spinner("Loading your progress... / กำลังโหลดความคืบหน้าของคุณ..."):
        meta, answers = load_remote_progress(session_id)

    order = None
    if meta and isinstance(meta.get("trial_order"), list) and len(meta["trial_order"]) == total:
        # Only trust a stored order that still matches the current manifest exactly.
        if sorted(meta["trial_order"]) == list(range(total)):
            order = list(meta["trial_order"])

    is_new = order is None
    if is_new:
        order = list(range(total))
        random.shuffle(order)
        answers = {}

    st.session_state.loaded_for = session_id
    st.session_state.session_id = session_id
    st.session_state.listener_id = user["email"]
    st.session_state.listener_name = user["name"]
    st.session_state.trial_order = order
    st.session_state.answers = answers
    st.session_state.pending_uploads = []
    st.session_state.started_at = (meta or {}).get("started_at") or _now()
    # Resuming participants skip the intro/demo and land back where they left off;
    # someone who already finished comes back to the thank-you/review page, not to a
    # clip index one past the end of the list.
    first_open = first_unanswered(order, answers)
    finished = first_open >= len(order)
    st.session_state.current_idx = 0 if finished else first_open
    if finished:
        st.session_state.stage = "done"
    else:
        st.session_state.stage = "intro" if (is_new and not answers) else "trial"
    st.session_state.resumed = bool(answers)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_session_meta():
    payload = {
        "session_id": st.session_state.session_id,
        "manifest_id": manifest_id(),
        "email": st.session_state.listener_id,
        "name": st.session_state.listener_name,
        "trial_order": st.session_state.trial_order,
        "total_trials": len(st.session_state.trial_order),
        "started_at": st.session_state.started_at,
        "updated_at": _now(),
    }
    ok, err = _upload_json(
        session_path(st.session_state.session_id),
        payload,
        f"session {st.session_state.session_id}",
    )
    return ok, err


def first_unanswered(order, answers) -> int:
    for i in range(len(order)):
        if i not in answers:
            return i
    return len(order)


def answered_count() -> int:
    return len(st.session_state.answers)


def save_answer(idx: int, row, mos: int, transcription: str):
    """Write one answer. Re-saving the same clip overwrites it, so edits just work."""
    record = {
        "session_id": st.session_state.session_id,
        "manifest_id": manifest_id(),
        "listener_id": st.session_state.listener_id,
        "listener_name": st.session_state.listener_name,
        "trial_index": idx,
        "utt_id": row["utt_id"],
        "token": row["token"],
        "mos": mos,
        "transcription": transcription.strip(),
        "timestamp": _now(),
    }
    existing = st.session_state.answers.get(idx)
    if existing:
        record["first_answered_at"] = existing.get("first_answered_at", existing.get("timestamp"))
        record["edited"] = True
    st.session_state.answers[idx] = record

    ok, err = _upload_json(
        trial_path(st.session_state.session_id, idx, row["utt_id"]),
        record,
        f"submission {st.session_state.session_id} #{idx}",
    )
    if not ok:
        st.session_state.pending_uploads.append(record)
        st.warning(
            f"Saved on this device, but uploading failed ({err}). The app will retry. / "
            "บันทึกไว้ในเครื่องแล้ว แต่อัปโหลดไม่สำเร็จ ระบบจะลองใหม่ให้อัตโนมัติ"
        )
    else:
        save_session_meta()
    return ok


def retry_pending():
    still_pending = []
    for record in st.session_state.get("pending_uploads", []):
        ok, _ = _upload_json(
            trial_path(record["session_id"], record["trial_index"], record["utt_id"]),
            record,
            f"retry submission {record['session_id']} #{record['trial_index']}",
        )
        if not ok:
            still_pending.append(record)
    st.session_state.pending_uploads = still_pending


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def intro_page():
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

There are **{total} short clips** in total.

**You do not have to finish in one sitting.** Your answers are saved automatically after
every clip. You can close the page at any time, sign back in with the same Google
account later, and carry on exactly where you left off. You can also **go back and
change** any earlier answer at any time.

Right after this page, you'll see one **example clip** to help you get familiar with
the task before the real ratings begin.

---

### ภาษาไทย
ขอบคุณที่เข้าร่วมการทดสอบการฟังในครั้งนี้

**สิ่งที่คุณจะทำ:** คุณจะได้ฟังคลิปเสียงภาษาไทยสั้น ๆ ทีละคลิป สำหรับแต่ละคลิป กรุณา:

1. **ฟังอย่างตั้งใจ** (แนะนำให้ใช้หูฟัง และอยู่ในที่เงียบ)
2. **ให้คะแนนความเป็นธรรมชาติ/ความเข้าใจง่าย** ของเสียงพูด ในระดับ 1-5
3. **พิมพ์สิ่งที่คุณได้ยินให้ตรงที่สุดเท่าที่จะทำได้** เป็นภาษาไทย แม้ว่าเสียงจะฟังดูไม่เป็นธรรมชาติ
   หรือคุณไม่แน่ใจทั้งหมด กรุณาพิมพ์คำตอบที่ใกล้เคียงที่สุดแทนการเว้นว่างไว้

มีคลิปเสียงทั้งหมด **{total} คลิป**

**คุณไม่จำเป็นต้องทำให้เสร็จในครั้งเดียว** ระบบจะบันทึกคำตอบของคุณโดยอัตโนมัติหลังจากทุกคลิป
คุณสามารถปิดหน้านี้เมื่อไรก็ได้ แล้วเข้าสู่ระบบด้วยบัญชี Google เดิมในภายหลัง
เพื่อทำต่อจากจุดที่ค้างไว้ และคุณยังสามารถ **ย้อนกลับไปแก้ไขคำตอบ** ก่อนหน้าได้ตลอดเวลา

หลังจากหน้านี้ คุณจะได้ฟัง **คลิปตัวอย่าง** หนึ่งคลิป เพื่อให้คุ้นเคยกับลักษณะงานก่อนเริ่มการให้คะแนนจริง
        """
    )
    if st.button("Continue / ดำเนินการต่อ", type="primary"):
        st.session_state.stage = "demo"
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
        # Persist the clip order only once they actually begin, so that merely signing
        # in (e.g. the conductor opening the app) doesn't register as a participant.
        save_session_meta()
        st.session_state.stage = "trial"
        st.rerun()


def trial_page():
    manifest = load_manifest()
    order = st.session_state.trial_order
    total = len(order)
    idx = max(0, min(st.session_state.current_idx, total - 1))
    st.session_state.current_idx = idx
    row = manifest.iloc[order[idx]]
    existing = st.session_state.answers.get(idx)

    done = answered_count()
    st.progress(
        done / total,
        text=f"{done} of {total} clips answered / ตอบแล้ว {done} จาก {total} คลิป",
    )
    st.caption(f"Clip {idx + 1} of {total} / คลิปที่ {idx + 1} จาก {total}")
    if existing:
        st.info(
            "You've already answered this clip. You can change your answer below. / "
            "คุณตอบคลิปนี้ไปแล้ว สามารถแก้ไขคำตอบด้านล่างได้"
        )

    audio_path = os.path.join(AUDIO_DIR, row["audio_filename"])
    with open(audio_path, "rb") as f:
        st.audio(f.read(), format="audio/wav")

    with st.form(key=f"trial_form_{idx}"):
        prev_mos = int(existing["mos"]) if existing else None
        mos = st.radio(
            "How natural / intelligible was this clip? / "
            "คลิปนี้ฟังเป็นธรรมชาติ/เข้าใจง่ายแค่ไหน?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: MOS_LABELS[x],
            index=([1, 2, 3, 4, 5].index(prev_mos) if prev_mos in (1, 2, 3, 4, 5) else None),
            horizontal=False,
        )
        transcription = st.text_area(
            "Type exactly what you heard (in Thai): / "
            "พิมพ์สิ่งที่คุณได้ยินให้ตรงที่สุด (เป็นภาษาไทย):",
            value=existing["transcription"] if existing else "",
            height=80,
        )
        col_prev, col_next = st.columns([1, 2])
        with col_prev:
            go_back = st.form_submit_button(
                "◀ Previous / ก่อนหน้า", disabled=(idx == 0), width="stretch"
            )
        with col_next:
            label = (
                "Save changes & continue / บันทึกการแก้ไขและไปต่อ"
                if existing
                else "Submit & continue / ส่งคำตอบและไปต่อ"
            )
            go_next = st.form_submit_button(label, type="primary", width="stretch")

    if not (go_back or go_next):
        return

    complete = mos is not None and transcription.strip()

    if go_back:
        # Never block going back; just keep whatever they'd filled in if it's usable.
        if complete:
            save_answer(idx, row, mos, transcription)
        st.session_state.current_idx = max(0, idx - 1)
        st.rerun()

    if not complete:
        if mos is None:
            st.error("Please select a rating before continuing. / กรุณาเลือกคะแนนก่อนไปต่อ")
        if not transcription.strip():
            st.error(
                "Please type what you heard before continuing (your best guess is fine). / "
                "กรุณาพิมพ์สิ่งที่คุณได้ยินก่อนไปต่อ (เดาที่ใกล้เคียงที่สุดก็ได้)"
            )
        return

    save_answer(idx, row, mos, transcription)
    if answered_count() >= total:
        st.session_state.stage = "done"
    else:
        st.session_state.current_idx = next_unanswered_from(idx)
    st.rerun()


def next_unanswered_from(idx: int) -> int:
    """Advance to idx+1, but skip past clips already answered (e.g. after editing)."""
    order = st.session_state.trial_order
    total = len(order)
    for i in range(idx + 1, total):
        if i not in st.session_state.answers:
            return i
    return min(idx + 1, total - 1)


def review_page():
    order = st.session_state.trial_order
    total = len(order)
    answers = st.session_state.answers

    st.title("Your answers / คำตอบของคุณ")
    st.markdown(
        f"You have answered **{len(answers)} of {total}** clips. Pick any clip below to "
        "listen again and change your answer.\n\n"
        f"คุณตอบไปแล้ว **{len(answers)} จาก {total}** คลิป เลือกคลิปใดก็ได้ด้านล่าง "
        "เพื่อฟังอีกครั้งและแก้ไขคำตอบ"
    )

    rows = []
    for i in range(total):
        rec = answers.get(i)
        rows.append(
            {
                "Clip / คลิป": i + 1,
                "Status / สถานะ": "✓ answered" if rec else "— not yet",
                "Rating / คะแนน": rec["mos"] if rec else None,
                "What you typed / สิ่งที่คุณพิมพ์": (rec["transcription"] if rec else ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    jump = st.selectbox(
        "Go to clip / ไปที่คลิป",
        options=list(range(1, total + 1)),
        index=min(st.session_state.current_idx, total - 1),
        format_func=lambda n: (
            f"Clip {n} - {'answered' if (n - 1) in answers else 'not answered yet'}"
        ),
    )
    if st.button("Open this clip / เปิดคลิปนี้", type="primary"):
        st.session_state.current_idx = jump - 1
        st.session_state.stage = "trial"
        st.rerun()


def break_page():
    done = answered_count()
    total = len(st.session_state.trial_order)
    st.title("Progress saved \U0001F4BE / บันทึกความคืบหน้าแล้ว")
    st.success(
        f"You've answered **{done} of {total}** clips, and all of it is saved.\n\n"
        f"คุณตอบไปแล้ว **{done} จาก {total}** คลิป และระบบบันทึกไว้ทั้งหมดแล้ว"
    )
    st.markdown(
        """
### English
Take as long a break as you like. To carry on, just open this same link again and sign
in with the same Google account -- you'll come back to exactly where you stopped.

### ภาษาไทย
พักได้นานเท่าที่ต้องการ เมื่อต้องการทำต่อ เพียงเปิดลิงก์เดิมนี้อีกครั้ง
และเข้าสู่ระบบด้วยบัญชี Google เดิม ระบบจะพากลับไปยังจุดที่คุณหยุดไว้
        """
    )
    if st.button("Continue now / ทำต่อเลย", type="primary"):
        st.session_state.stage = "trial"
        st.rerun()


def done_page():
    retry_pending()
    total = len(st.session_state.trial_order)
    st.title("Thank you! \U0001F389 / ขอบคุณที่เข้าร่วม! \U0001F389")
    st.markdown(
        f"You completed all **{total}** clips. Your responses have been recorded.\n\n"
        f"คุณทำแบบทดสอบครบทั้งหมด **{total}** คลิปแล้ว คำตอบของคุณถูกบันทึกเรียบร้อยแล้ว"
    )
    st.markdown(
        "If you'd like to change anything, you can still review and edit your answers. / "
        "หากต้องการแก้ไข คุณยังสามารถตรวจทานและแก้ไขคำตอบได้"
    )
    if st.button("Review & edit my answers / ตรวจทานและแก้ไขคำตอบ"):
        st.session_state.stage = "review"
        st.rerun()

    if st.session_state.pending_uploads:
        st.warning(
            f"{len(st.session_state.pending_uploads)} response(s) could not be uploaded "
            "automatically. Please use the download button below and send the file to the "
            "test organizer as a backup.\n\n"
            f"มีคำตอบ {len(st.session_state.pending_uploads)} รายการที่ไม่สามารถอัปโหลดโดยอัตโนมัติได้ "
            "กรุณากดปุ่มดาวน์โหลดด้านล่างแล้วส่งไฟล์ให้ผู้จัดการทดสอบเป็นข้อมูลสำรอง"
        )

    ordered = [st.session_state.answers[i] for i in sorted(st.session_state.answers)]
    df = pd.DataFrame(ordered)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download my responses (CSV backup) / ดาวน์โหลดคำตอบของฉัน (ไฟล์สำรอง CSV)",
        data=csv_bytes,
        file_name=f"responses_{st.session_state.session_id}.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------------------
# Conductor view
# --------------------------------------------------------------------------------------
def monitor_page():
    st.title("Participant progress / ความคืบหน้าของผู้เข้าร่วม")
    total = len(load_manifest())
    prefix = f"{manifest_id()}/"

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Refresh / รีเฟรช"):
            st.cache_data.clear()
            st.rerun()
    with col_b:
        st.caption(f"Current test set: `{manifest_id()}` - {total} clips per participant")

    if get_api() is None:
        st.error("Results storage is not configured (HF_TOKEN / RESULTS_REPO_ID missing).")
        return

    files = [f for f in list_repo_files_cached() if f.startswith(prefix)]
    if not files:
        st.info("No participants have started this test set yet.")
        return

    by_session = {}
    for path in files:
        parts = path[len(prefix):].split("/")
        if len(parts) != 2:
            continue
        by_session.setdefault(parts[0], []).append(parts[1])

    rows = []
    for session_id, names in sorted(by_session.items()):
        answered = sum(1 for n in names if n != SESSION_FILENAME and n.endswith(".json"))
        meta = load_session_meta_cached(f"{prefix}{session_id}/{SESSION_FILENAME}") or {}
        rows.append(
            {
                "Participant": meta.get("name") or session_id,
                "Email": meta.get("email", ""),
                "Answered": answered,
                "of": total,
                "Progress": answered / total if total else 0.0,
                "Started (UTC)": (meta.get("started_at") or "")[:19].replace("T", " "),
                "Last activity (UTC)": (meta.get("updated_at") or "")[:19].replace("T", " "),
            }
        )

    df = pd.DataFrame(rows).sort_values("Progress", ascending=False)
    finished = int((df["Answered"] >= total).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Participants started", len(df))
    c2.metric("Finished", finished)
    c3.metric("Total ratings collected", int(df["Answered"].sum()))

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Progress": st.column_config.ProgressColumn(
                "Progress", min_value=0.0, max_value=1.0, format="%.0f%%"
            )
        },
    )
    st.caption(
        "Note: 'Last activity' updates each time a participant saves an answer. "
        "The listing is cached for 30 seconds -- use Refresh to force an update."
    )


# --------------------------------------------------------------------------------------
def sidebar(user: dict, is_admin: bool):
    with st.sidebar:
        st.markdown(f"**{user['name']}**")
        st.caption(user["email"])

        if st.session_state.get("stage") in ("trial", "review", "done", "break"):
            total = len(st.session_state.trial_order)
            done = answered_count()
            st.progress(done / total, text=f"{done}/{total}")
            st.markdown("---")
            if st.session_state.stage != "trial":
                if st.button("Back to the clips / กลับไปที่คลิป", width="stretch"):
                    st.session_state.stage = "trial"
                    st.rerun()
            if st.session_state.stage != "review":
                if st.button(
                    "Review / edit answers / ตรวจทานและแก้ไข", width="stretch"
                ):
                    st.session_state.stage = "review"
                    st.rerun()
            if st.button("Take a break / พักก่อน", width="stretch"):
                st.session_state.stage = "break"
                st.rerun()

        st.markdown("---")
        if is_admin:
            if st.session_state.get("view") == "monitor":
                if st.button("← Back to the test", width="stretch"):
                    st.session_state.view = "test"
                    st.rerun()
            else:
                if st.button("\U0001F4CA Monitor participants", width="stretch"):
                    st.session_state.view = "monitor"
                    st.rerun()

        if st.button("Sign out / ออกจากระบบ", width="stretch"):
            do_logout()


def main():
    user = login_gate()

    email = user["email"]
    is_admin = email in ADMIN_EMAILS

    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS and not is_admin:
        st.error(
            f"**{email}** is not on the participant list for this test. If you think this "
            "is a mistake, please contact the test organizer.\n\n"
            f"**{email}** ไม่อยู่ในรายชื่อผู้เข้าร่วมการทดสอบนี้ "
            "หากคิดว่าเป็นข้อผิดพลาด กรุณาติดต่อผู้จัดการทดสอบ"
        )
        if st.button("Sign out / ออกจากระบบ"):
            do_logout()
        st.stop()

    ensure_session_loaded(user)
    sidebar(user, is_admin)

    if is_admin and st.session_state.get("view") == "monitor":
        monitor_page()
        return

    stage = st.session_state.get("stage", "intro")

    if stage == "intro":
        intro_page()
    elif stage == "demo":
        demo_page()
    elif stage == "review":
        review_page()
    elif stage == "break":
        break_page()
    elif stage == "done":
        done_page()
    else:
        if st.session_state.get("resumed"):
            st.session_state.resumed = False
            st.toast("Welcome back -- resuming where you left off. / ยินดีต้อนรับกลับมา")
        trial_page()


if __name__ == "__main__":
    main()
