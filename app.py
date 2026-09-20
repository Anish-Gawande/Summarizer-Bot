import io

import streamlit as st
from pypdf import PdfReader
from transformers import AutoTokenizer, pipeline

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MODEL_MAX_TOKENS = 1024
CHUNK_TOKENS = 850  # leave headroom for special tokens

st.set_page_config(page_title="Summarizer Bot", layout="wide")


@st.cache_resource(show_spinner="Loading the summarization model (first run only)...")
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    summarizer = pipeline(
        "summarization",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        device=-1,  # CPU: Streamlit Community Cloud has no GPU
    )
    return summarizer, tokenizer


summarizer, tokenizer = load_summarizer()


def chunk_text(text, max_tokens=CHUNK_TOKENS):
    """Split text into pieces the model can actually accept."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    for start in range(0, len(ids), max_tokens):
        piece = ids[start : start + max_tokens]
        yield tokenizer.decode(piece, skip_special_tokens=True)


def _summarize_one(chunk):
    n_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
    if n_tokens < 40:
        return chunk  # too short to summarize meaningfully
    max_len = max(40, min(150, int(n_tokens * 0.45)))
    min_len = max(20, min(40, max_len - 15))
    out = summarizer(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )
    return out[0]["summary_text"].strip()


def text_summary(text):
    text = " ".join(text.split())
    if not text:
        return ""

    chunks = list(chunk_text(text))
    partials = []
    progress = st.progress(0.0, text="Summarizing...")
    for i, chunk in enumerate(chunks, start=1):
        partials.append(_summarize_one(chunk))
        progress.progress(i / len(chunks), text=f"Summarizing chunk {i} of {len(chunks)}...")
    progress.empty()

    combined = " ".join(partials)

    # Second pass so a long document collapses into one coherent summary
    if len(chunks) > 1:
        second_pass = [_summarize_one(c) for c in chunk_text(combined)]
        combined = " ".join(second_pass)

    return combined


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    parts = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            parts.append(content)
    return "\n".join(parts)


def parse_video_id(url):
    url = url.strip()
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("/")[0]
    if "/shorts/" in url:
        return url.split("/shorts/")[1].split("?")[0].split("/")[0]
    if "/embed/" in url:
        return url.split("/embed/")[1].split("?")[0].split("/")[0]
    return None


def extract_transcript(video_url, language="en"):
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = parse_video_id(video_url)
    if not video_id:
        st.warning("That doesn't look like a valid YouTube URL.")
        return None

    try:
        # youtube-transcript-api >= 1.0 uses an instance API;
        # older versions expose the static get_transcript().
        if hasattr(YouTubeTranscriptApi, "fetch"):
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=[language, "en"])
            return " ".join(snippet.text for snippet in fetched)
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=[language, "en"])
        return " ".join(entry["text"] for entry in entries)
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the user
        st.error(f"Could not fetch the transcript: {exc}")
        st.caption(
            "YouTube frequently blocks requests coming from cloud datacenter IPs, "
            "so this can fail on Streamlit Cloud even when it works locally. "
            "Paste the transcript into the 'Summarize Text' tab as a fallback."
        )
        return None


choice = st.sidebar.selectbox(
    "Select your choice",
    ["Summarize Text", "Summarize Document", "Summarize YouTube Video"],
)

if choice == "Summarize Text":
    st.subheader("Summarize Text")
    input_text = st.text_area("Enter your text here", height=250)
    if st.button("Summarize Text", disabled=not input_text.strip()):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Your Input Text**")
            st.info(input_text)
        with col2:
            st.markdown("**Summary Result**")
            st.success(text_summary(input_text))

elif choice == "Summarize Document":
    st.subheader("Summarize Document")
    input_file = st.file_uploader("Upload your document here", type=["pdf"])
    if input_file and st.button("Summarize Document"):
        text = extract_text_from_pdf(input_file)
        if not text.strip():
            st.error(
                "No selectable text found. This looks like a scanned PDF, "
                "which needs OCR before it can be summarized."
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Extracted Text**")
                st.text_area("extracted", text, height=400, label_visibility="collapsed")
            with col2:
                st.markdown("**Summary Result**")
                st.success(text_summary(text))

elif choice == "Summarize YouTube Video":
    st.subheader("Summarize YouTube Video")
    video_url = st.text_input("Enter YouTube Video URL")
    language = st.selectbox("Select language", ["en", "hi", "mr"])
    if video_url and st.button("Summarize YouTube Video"):
        with st.spinner("Fetching transcript..."):
            transcript = extract_transcript(video_url, language)
        if transcript:
            st.markdown("**Summary Result**")
            st.success(text_summary(transcript))
            with st.expander("Show full transcript"):
                st.write(transcript)
