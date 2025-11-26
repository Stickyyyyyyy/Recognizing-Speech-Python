import os
import re
import json
import threading
import time
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import speech_recognition as sr
from rapidfuzz import process, fuzz

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

try:
    from pydub import AudioSegment
    from pydub.utils import which as pydub_which
    HAVE_PYDUB = True
except ImportError:
    HAVE_PYDUB = False

    def pydub_which(name: str):
        return None


DICT_PATH = Path("words.txt")
FUZZY_CUTOFF = 85
DEFAULT_LANGUAGE = "ru-RU"
DEFAULT_TRIM_THRESHOLD = 400
DEFAULT_TRIM_PAD_MS = 100
VU_MAX = 3000

ENABLE_AI = True

LEARN_DB_PATH = Path("learned_punct.json")
LEARN_LOG_PATH = Path("learning_log.jsonl")


def load_learn_db():
    if LEARN_DB_PATH.exists():
        try:
            return json.loads(LEARN_DB_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"ru": {}, "en": {}}
    return {"ru": {}, "en": {}}


LEARN_DB = load_learn_db()


def save_learn_db():
    try:
        LEARN_DB_PATH.write_text(
            json.dumps(LEARN_DB, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def word_punct_pairs(text: str):
    """
    Split text into (word, trailing_punctuation) pairs, per whitespace token.
    Example: "слово, тест..." -> [("слово", ","), ("тест", "...")]
    """
    tokens = text.split()
    pairs = []
    for tok in tokens:
        m = re.match(r"(.+?)([.,!?…]+)?$", tok)
        if m:
            word = m.group(1)
            punct = m.group(2) or ""
        else:
            word, punct = tok, ""
        pairs.append((word, punct))
    return pairs


def update_learn_db(lang: str, ai_text: str, ref_text: str):
    """
    Compare AI-fixed text with reference text, word-by-word, and update
    learned punctuation preferences.

    - If AI put a comma but reference did not -> comma score--
    - If AI missed a comma but reference has it -> comma score++
    (similarly for period and ellipsis)
    """
    ai_pairs = word_punct_pairs(ai_text)
    ref_pairs = word_punct_pairs(ref_text)
    n = min(len(ai_pairs), len(ref_pairs))
    db_lang = LEARN_DB.setdefault(lang, {})

    for i in range(n):
        w_ai, p_ai = ai_pairs[i]
        w_ref, p_ref = ref_pairs[i]
        if not w_ref:
            continue

        key = w_ref.lower()
        entry = db_lang.setdefault(key, {"comma": 0, "period": 0, "ellipsis": 0})

        if ("," in p_ai) and ("," not in p_ref):
            entry["comma"] -= 1
        elif ("," not in p_ai) and ("," in p_ref):
            entry["comma"] += 1

        if any(c in p_ai for c in ".!?") and not any(c in p_ref for c in ".!?"):
            entry["period"] -= 1
        elif not any(c in p_ai for c in ".!?") and any(c in p_ref for c in ".!?"):
            entry["period"] += 1

        if "..." in p_ai and "..." not in p_ref:
            entry["ellipsis"] -= 1
        elif "..." not in p_ai and "..." in p_ref:
            entry["ellipsis"] += 1


def get_punct_prefs(lang: str, word: str):
    """
    Return learned punctuation preferences for a given word and language.
    Structure: {"comma": int, "period": int, "ellipsis": int}
    Negative values mean "tend to avoid", positive mean "tend to add".
    """
    db_lang = LEARN_DB.get(lang, {})
    return db_lang.get(word.lower(), {"comma": 0, "period": 0, "ellipsis": 0})


def load_dictionary(path: Path):
    if path.exists():
        words = [
            w.strip().lower()
            for w in path.read_text(encoding="utf-8").splitlines()
            if w.strip()
        ]
    else:
        words = [
            "hello", "world", "computer", "program",
            "dictionary", "banana", "apple", "test",
        ]
    return words, set(words)


DICT_LIST, DICT_SET = load_dictionary(DICT_PATH)


def dict_match(token: str, cutoff: int = FUZZY_CUTOFF):
    t = token.lower()
    if t in DICT_SET:
        return t
    match = process.extractOne(t, DICT_LIST, scorer=fuzz.WRatio)
    return match[0] if match and match[1] >= cutoff else t


def trim_silence(
    samples: np.ndarray,
    threshold: int = DEFAULT_TRIM_THRESHOLD,
    pad_samples: int = 1600,
) -> np.ndarray:
    if samples.size == 0:
        return samples
    abs_sig = np.abs(samples.astype(np.int32))
    voiced = np.where(abs_sig > threshold)[0]
    if voiced.size == 0:
        return samples
    start = max(0, voiced[0] - pad_samples)
    end = min(samples.size, voiced[-1] + pad_samples + 1)
    return samples[start:end]


def normalize_int16(samples: np.ndarray, peak: int = 30000) -> np.ndarray:
    if samples.size == 0:
        return samples
    m = int(np.max(np.abs(samples.astype(np.int32))))
    if m <= 0:
        return samples
    scale = float(peak) / float(m)
    return np.clip(
        samples.astype(np.float32) * scale, -32768, 32767
    ).astype(np.int16)


def resample_to_16k(samples: np.ndarray, src_sr: int) -> np.ndarray:
    dst_sr = 16000
    if src_sr == dst_sr or samples.size == 0:
        return samples.astype(np.int16, copy=False)
    src_len = samples.shape[0]
    t_src = np.linspace(0.0, 1.0, num=src_len, endpoint=False, dtype=np.float32)
    dst_len = int(np.round(src_len * (dst_sr / float(src_sr))))
    t_dst = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float32)
    interp = np.interp(t_dst, t_src, samples.astype(np.float32))
    return np.clip(interp, -32768, 32767).astype(np.int16)


def numpy_to_audio_data_16k(samples: np.ndarray, src_sr_hz: int) -> sr.AudioData:
    s16k = resample_to_16k(samples, src_sr_hz)
    return sr.AudioData(s16k.tobytes(), 16000, 2)


def recognize_google_best(audio_data: sr.AudioData, language: str) -> str:
    r = sr.Recognizer()
    try:
        result = r.recognize_google(audio_data, language=language, show_all=True)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise RuntimeError(f"Network/API error: {e}")

    if isinstance(result, dict) and "alternative" in result and result["alternative"]:
        alts = result["alternative"]
        best = None
        best_conf = -1.0
        for alt in alts:
            conf = float(alt.get("confidence", -1.0))
            if conf > best_conf and "transcript" in alt:
                best_conf = conf
                best = alt["transcript"]
        if best is None:
            best = alts[0].get("transcript", "")
        return best.strip()

    if isinstance(result, str):
        return result.strip()
    return ""


def detect_lang_from_text(text: str) -> str:
    """Very rough: Russian if any Cyrillic, else English."""
    for ch in text:
        if "\u0400" <= ch <= "\u04FF":
            return "ru"
    return "en"


def detect_pauses(
    samples: np.ndarray,
    sr: int,
    silence_threshold: int = DEFAULT_TRIM_THRESHOLD,
    min_pause_ms: int = 200,
):
    """
    Find pauses (silence segments) inside the audio.

    Returns list of (relative_position_0_to_1, duration_ms).
    """
    if samples.size == 0:
        return []
    abs_sig = np.abs(samples)
    silence = abs_sig < silence_threshold

    pauses = []
    in_silence = False
    start_idx = 0
    n = len(silence)

    for i, s in enumerate(silence):
        if s and not in_silence:
            in_silence = True
            start_idx = i
        elif not s and in_silence:
            in_silence = False
            dur_ms = (i - start_idx) / float(sr) * 1000.0
            if dur_ms >= min_pause_ms:
                center = (start_idx + i) / 2.0
                pos = center / n
                pauses.append((pos, dur_ms))

    if in_silence:
        end = n
        dur_ms = (end - start_idx) / float(sr) * 1000.0
        if dur_ms >= min_pause_ms:
            center = (start_idx + end) / 2.0
            pos = center / n
            pauses.append((pos, dur_ms))

    return pauses


def capitalize_sentences(text: str) -> str:
    result = []
    new_sentence = True
    for ch in text:
        if new_sentence and ch.isalpha():
            result.append(ch.upper())
            new_sentence = False
        else:
            result.append(ch)
        if ch in ".!?":
            new_sentence = True
    return "".join(result)


def apply_pause_punctuation(
    tokens,
    pauses,
    lang: str,
    comma_ms: int = 350,
    ellipsis_ms: int = 900,
    period_ms: int = 1500,
):
    """
    Insert commas / ellipses / full stops into tokens based on pause positions
    and learned preferences.

    tokens: list of words (possibly with punctuation)
    pauses: list of (relative_pos, duration_ms)
    """
    if not pauses or not tokens:
        return tokens

    n = len(tokens)
    used = set()

    for pos, dur in pauses:
        idx = int(round(pos * (n - 1)))
        if idx < 0 or idx >= n or idx in used:
            continue
        used.add(idx)

        word = tokens[idx]
        if word.endswith((".", "!", "?", ",", ";", ":", "...")):
            continue

        if dur >= period_ms:
            punct = "."
        elif dur >= ellipsis_ms:
            punct = "..."
        elif dur >= comma_ms:
            punct = ","
        else:
            continue

        base = re.sub(r"[^\w\u0400-\u04FF]", "", word, flags=re.UNICODE)
        prefs = get_punct_prefs(lang, base) if base else {"comma": 0, "period": 0, "ellipsis": 0}

        if punct == "," and prefs["comma"] < 0:
            continue
        if punct == "." and prefs["period"] < 0:
            continue
        if punct == "..." and prefs["ellipsis"] < 0:
            continue

        tokens[idx] = word + punct

    return tokens


def fix_english_paragraph(text: str, pauses=None) -> str:
    """
    Simple English fixer:
    - normalizes spaces
    - fixes I / some contractions
    - converts fillers (uh, um, eh, mm) to "..."
    - uses audio pauses + learned prefs for commas/ellipses/full stops
    - capitalizes sentence starts, ensures final punctuation
    """
    text = text.strip()
    if not text:
        return text

    text = re.sub(r"\s+", " ", text)
    tokens = text.split(" ")

    corrections = {
        "im": "I'm",
        "i'm": "I'm",
        "dont": "don't",
        "cant": "can't",
        "wont": "won't",
        "lets": "let's",
        "ive": "I've",
        "id": "I'd",
        "youre": "you're",
        "theres": "there's",
        "isnt": "isn't",
        "arent": "aren't",
        "shouldnt": "shouldn't",
        "couldnt": "couldn't",
        "wouldnt": "wouldn't",
        "didnt": "didn't",
    }
    conj_for_commas = {"but", "however", "although", "though", "because", "so"}
    fillers_en = {"uh", "um", "er", "eh", "hmm", "mm", "uhm", "uhhh", "mmm"}

    new_tokens = []
    for i, tok in enumerate(tokens):
        m = re.match(r"([A-Za-z']+)([^A-Za-z']*)", tok)
        if m:
            word, punct = m.group(1), m.group(2)
        else:
            word, punct = tok, ""

        low = word.lower()

        if low in fillers_en:
            if new_tokens and not new_tokens[-1].endswith((".", "!", "?", "...", ",")):
                new_tokens[-1] = new_tokens[-1] + "..."
            continue

        if low == "i":
            word = "I"
        elif low in corrections:
            word = corrections[low]

        if i > 0 and low in conj_for_commas:
            if new_tokens:
                prev = new_tokens[-1]
                if not prev.endswith((",", ".", "!", "?", ";", ":")):
                    new_tokens[-1] = prev + ","

        new_tokens.append(word + punct)

    if pauses:
        new_tokens = apply_pause_punctuation(new_tokens, pauses, "en")

    text = " ".join(new_tokens)
    text = capitalize_sentences(text)
    text = text.rstrip()
    if text and text[-1] not in ".!?…":
        text += "."
    return text


def fix_russian_paragraph(text: str, pauses=None) -> str:
    """
    Simple Russian fixer:
    - normalizes spaces
    - treats fillers like «ну», «э», «эээ», «мм» as pauses → "..."
    - text-based commas before «но», «а», «однако», etc.
    - audio-based commas / ellipses / full stops with learning
    - capitalizes sentence starts, ensures final punctuation
    """
    text = text.strip()
    if not text:
        return text

    text = re.sub(r"\s+", " ", text)
    tokens = text.split(" ")

    conj_ru = {"но", "а", "однако", "зато", "поэтому"}
    fillers_ru = {"э", "ээ", "эээ", "мм", "м-м", "ну", "типо", "типа"}

    new_tokens = []
    for i, tok in enumerate(tokens):
        base = tok.strip(",.?!:;…").lower()

        if base in fillers_ru:
            if new_tokens and not new_tokens[-1].endswith((".", "!", "?", "...", ",")):
                new_tokens[-1] = new_tokens[-1] + "..."
            continue

        if i > 0 and base in conj_ru:
            if new_tokens:
                prev = new_tokens[-1]
                if not prev.endswith((",", ".", "!", "?", ";", ":")):
                    new_tokens[-1] = prev + ","

        new_tokens.append(tok)

    if pauses:
        new_tokens = apply_pause_punctuation(new_tokens, pauses, "ru")

    text = " ".join(new_tokens)
    text = capitalize_sentences(text)
    text = text.rstrip()
    if text and text[-1] not in ".!?…":
        text += "."
    return text


def ai_postprocess(
    asr_text: str,
    dict_words,
    language: str | None = None,
    pauses=None,
) -> dict:
    """
    Local rule-based 'AI':
    - uses pauses + text to decide commas, dots, ellipses
    - handles filler sounds (uh, э-э, мм) as prosodic pauses
    - fixes capitalization and some simple errors.
    Returns only corrected_text (no intent/entities).
    """
    corrected = asr_text.strip()
    if not corrected:
        return {"corrected_text": ""}

    lang = language or detect_lang_from_text(corrected)

    if lang == "ru":
        corrected = fix_russian_paragraph(corrected, pauses)
    else:
        corrected = fix_english_paragraph(corrected, pauses)

    return {"corrected_text": corrected}


if HAVE_PYDUB:
    AudioSegment.converter = pydub_which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    AudioSegment.ffprobe = pydub_which("ffprobe") or r"C:\ffmpeg\bin\ffprobe.exe"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speech to Text")
        self.geometry("900x720")
        self.minsize(820, 600)

        self.recording = False
        self.chunks: list[np.ndarray] = []
        self.stream = None
        self.level_q: queue.Queue[float] = queue.Queue()
        self.audio_level = 0.0

        self._build_ui()
        self._populate_devices()
        self._tick_vu()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Device:").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_map: dict[str, int] = {}
        self.device_cb = ttk.Combobox(
            top, textvariable=self.device_var, state="readonly", width=46
        )
        self.device_cb.pack(side="left", padx=6)
        ttk.Button(top, text="Refresh", command=self._populate_devices).pack(
            side="left", padx=6
        )

        thr_frame = ttk.Frame(self)
        thr_frame.pack(fill="x", **pad)

        ttk.Label(thr_frame, text="Silence threshold:").pack(side="left")

        self.thr_var = tk.IntVar(value=DEFAULT_TRIM_THRESHOLD)
        self.thr_label = ttk.Label(thr_frame, text=str(DEFAULT_TRIM_THRESHOLD))
        self.thr_label.pack(side="right")

        self.thr_scale = ttk.Scale(
            thr_frame,
            from_=100,
            to=1500,
            orient="horizontal",
            variable=self.thr_var,
            command=lambda v: self.thr_label.config(text=str(int(float(v)))),
        )
        self.thr_scale.set(DEFAULT_TRIM_THRESHOLD)
        self.thr_scale.pack(side="left", fill="x", expand=True, padx=8)

        vu_frame = ttk.Frame(self)
        vu_frame.pack(fill="x", **pad)
        ttk.Label(vu_frame, text="Mic level:").pack(side="left")
        self.vu = ttk.Progressbar(vu_frame, maximum=VU_MAX)
        self.vu.pack(side="left", fill="x", expand=True, padx=8)

        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        self.start_btn = ttk.Button(btns, text="Start (S)", command=self.start_recording)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            btns, text="Stop (E)", command=self.stop_recording, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=6)
        self.open_btn = ttk.Button(
            btns, text="Open audio file…", command=self.open_file_and_process
        )
        self.open_btn.pack(side="left", padx=6)
        self.learn_btn = ttk.Button(
            btns, text="Learn from reference", command=self.learn_from_reference
        )
        self.learn_btn.pack(side="left", padx=6)

        out = ttk.Frame(self)
        out.pack(fill="both", expand=True, **pad)

        self.asr_box = self._add_box(out, "Raw ASR")
        self.dict_box = self._add_box(out, "Dict-mapped")
        self.ai_box = self._add_box(out, "AI-fixed (learning)")
        self.ref_box = self._add_box(out, "Reference (correct text)")

        self.bind_all("<Key>", self._hotkeys)

    def _add_box(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill="both", expand=True, pady=6)
        box = scrolledtext.ScrolledText(frame, height=4, wrap="word")
        box.pack(fill="both", expand=True)
        box.configure(font=("Consolas", 11))
        return box

    def _populate_devices(self):
        self.device_cb["values"] = ()
        self.device_map.clear()
        devices = sd.query_devices()
        inputs = [
            (i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0
        ]
        labels: list[str] = []
        for i, d in inputs:
            sr_hz = int(d.get("default_samplerate", 16000) or 16000)
            label = f"[{i}] {d['name']} — {sr_hz} Hz"
            labels.append(label)
            self.device_map[label] = i
        if not labels:
            messagebox.showerror("No inputs", "No microphone devices found.")
            return
        self.device_cb["values"] = labels
        self.device_var.set(labels[0])

    def start_recording(self):
        if self.recording:
            return
        label = self.device_var.get()
        if label not in self.device_map:
            messagebox.showerror("Device", "Choose an input device first.")
            return
        self.device_idx = self.device_map[label]
        dev_info = sd.query_devices(self.device_idx)
        self.sr_hz = int(dev_info.get("default_samplerate", 16000) or 16000)

        self.chunks = []
        self.recording = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.asr_box.delete("1.0", "end")
        self.dict_box.delete("1.0", "end")
        self.ai_box.delete("1.0", "end")
        self.ref_box.delete("1.0", "end")

        def callback(indata, frames, time_info, status):
            if status:
                pass
            if self.recording:
                self.chunks.append(indata.copy())
                rms = float(
                    np.sqrt(np.mean(indata.astype(np.float32) ** 2))
                ) * 32768.0
                try:
                    self.level_q.put_nowait(min(rms, VU_MAX))
                except queue.Full:
                    pass

        def run_stream():
            try:
                with sd.InputStream(
                    samplerate=self.sr_hz,
                    channels=1,
                    dtype="int16",
                    device=self.device_idx,
                    callback=callback,
                    blocksize=0,
                    latency="low",
                ):
                    while self.recording:
                        time.sleep(0.05)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Audio error", str(e)))
                self.recording = False

        self.stream_thread = threading.Thread(target=run_stream, daemon=True)
        self.stream_thread.start()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        threading.Thread(target=self._process_audio, daemon=True).start()

    def learn_from_reference(self):
        ai_text = self.ai_box.get("1.0", "end").strip()
        ref_text = self.ref_box.get("1.0", "end").strip()
        if not ai_text or not ref_text:
            messagebox.showwarning(
                "Learn",
                "Provide both AI-fixed text and reference (correct) text."
            )
            return

        lang = detect_lang_from_text(ref_text or ai_text)
        update_learn_db(lang, ai_text, ref_text)
        save_learn_db()

        try:
            with LEARN_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"lang": lang, "ai_text": ai_text, "ref_text": ref_text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            pass

        messagebox.showinfo("Learn", "Saved correction and updated punctuation preferences.")

    def open_file_and_process(self):
        if self.recording:
            self.stop_recording()

        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=(
                ("Audio files", "*.wav *.flac *.aiff *.aif *.mp3 *.m4a *.ogg *.wma"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        threading.Thread(
            target=self._process_audio_file, args=(path,), daemon=True
        ).start()

    def _process_audio_file(self, path: str):
        self._set_text(self.asr_box, f"[Processing file] {os.path.basename(path)}")
        self._set_text(self.dict_box, "")
        self._set_text(self.ai_box, "")
        self._set_text(self.ref_box, "")

        ext = Path(path).suffix.lower()
        recognizer = sr.Recognizer()

        try:
            if ext in (".wav", ".flac", ".aiff", ".aif"):
                with sr.AudioFile(path) as source:
                    audio_data = recognizer.record(source)
                raw16 = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
            else:
                if not HAVE_PYDUB:
                    self._set_text(
                        self.asr_box,
                        "[File error] This format requires 'pydub' and ffmpeg installed.",
                    )
                    return
                sound = AudioSegment.from_file(path)
                sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                raw16 = sound.raw_data
                audio_data = sr.AudioData(raw16, 16000, 2)
        except Exception as e:
            self._set_text(self.asr_box, f"[File error] {e}")
            return

        samples = np.frombuffer(raw16, dtype=np.int16)
        pauses = detect_pauses(samples, 16000)

        asr_lang = DEFAULT_LANGUAGE
        try:
            text = recognize_google_best(audio_data, language=asr_lang)
        except RuntimeError as e:
            self._set_text(self.asr_box, f"[ASR error] {e}")
            return

        if not text:
            self._set_text(self.asr_box, "🙈 Didn’t catch that.")
            return

        self._set_text(self.asr_box, text)
        tokens = [t for t in text.lower().split() if t]
        mapped = [dict_match(t) for t in tokens]
        self._set_text(self.dict_box, " ".join(mapped))

        lang_hint = detect_lang_from_text(text)
        ai = ai_postprocess(text, list(DICT_SET), language=lang_hint, pauses=pauses)
        self._set_text(self.ai_box, ai.get("corrected_text", text))

    def _process_audio(self):
        if not self.chunks:
            self._set_text(self.asr_box, "No audio captured.")
            return

        samples = np.concatenate(self.chunks, axis=0).flatten()

        pad_samples = int(self.sr_hz * (DEFAULT_TRIM_PAD_MS / 1000.0))
        threshold = int(self.thr_scale.get())
        samples = trim_silence(samples, threshold=threshold, pad_samples=pad_samples)

        pauses = detect_pauses(samples, self.sr_hz, silence_threshold=threshold)

        samples = normalize_int16(samples, peak=30000)
        audio_data = numpy_to_audio_data_16k(samples, self.sr_hz)

        asr_lang = DEFAULT_LANGUAGE
        try:
            text = recognize_google_best(audio_data, language=asr_lang)
        except RuntimeError as e:
            self._set_text(self.asr_box, f"[ASR error] {e}")
            return

        if not text:
            self._set_text(self.asr_box, "🙈 Didn’t catch that.")
            return

        self._set_text(self.asr_box, text)

        tokens = [t for t in text.lower().split() if t]
        mapped = [dict_match(t) for t in tokens]
        self._set_text(self.dict_box, " ".join(mapped))

        lang_hint = detect_lang_from_text(text)
        ai = ai_postprocess(text, list(DICT_SET), language=lang_hint, pauses=pauses)
        self._set_text(self.ai_box, ai.get("corrected_text", text))

    def _set_text(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="normal")

    def _tick_vu(self):
        try:
            while True:
                val = self.level_q.get_nowait()
                self.vu["value"] = val
        except queue.Empty:
            pass
        self.after(50, self._tick_vu)

    def _hotkeys(self, event):
        k = event.keysym.lower()
        if k == "s":
            self.start_recording()
        elif k == "e":
            self.stop_recording()
        elif k == "q":
            self.on_close()

    def on_close(self):
        self.recording = False
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
