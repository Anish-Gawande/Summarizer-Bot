import os
import streamlit as st
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfReader
from transformers import pipeline

st.set_page_config(layout="wide")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

def text_summary(text, maxlen=130, minlen=30):
    result = summarizer(text, max_length=maxlen, min_length=minlen, do_sample=False)
    return result[0]['summary_text']

def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def download_youtube_video(video_url):
    os.makedirs("videos", exist_ok=True)
    yt = YouTube(video_url)
    video_stream = yt.streams.filter(file_extension="mp4").first()
    video_path = os.path.join("videos", f"{yt.title}.mp4")
    video_stream.download(output_path="videos", filename=yt.title)
    return video_path

def extract_transcript(video_url, language="en"):
    try:
        video_id = None
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be" in video_url:
            video_id = video_url.split("/")[-1]
        if video_id:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            return " ".join([entry['text'] for entry in transcript])
        else:
            st.warning("Invalid YouTube URL format.")
            return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document", "Summarize YouTube Video"])

if choice == "Summarize Text":
    st.subheader("Summarize Text")
    input_text = st.text_area("Enter your text here")
    if input_text.strip():
        if st.button("Summarize Text"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                summary = text_summary(input_text)
                st.success(summary)

elif choice == "Summarize Document":
    st.subheader("Summarize Document")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            text = extract_text_from_pdf("doc_file.pdf")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Extracted Text**")
                st.info(text)
            with col2:
                st.markdown("**Summary Result**")
                doc_summary = text_summary(text)
                st.success(doc_summary)

elif choice == "Summarize YouTube Video":
    st.subheader("Summarize YouTube Video")
    video_url = st.text_input("Enter YouTube Video URL")
    language = st.selectbox("Select language", ["en", "mr", "hi"])
    if video_url:
        if st.button("Summarize YouTube Video"):
            st.info("Extracting transcript...")
            transcript = extract_transcript(video_url, language)
            if transcript:
                st.info("Summarizing transcript...")
                video_summary = text_summary(transcript)
                st.markdown("**Summary Result**")
                st.success(video_summary)
            else:
                st.warning("Transcript not available or could not be fetched.")
