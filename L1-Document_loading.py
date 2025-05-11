# %% [markdown]
# # Document Loading

# %% [markdown]
# ## Note to students.
# During periods of high load you may find the notebook unresponsive. It may appear to execute a cell, update the completion number in brackets [#] at the left of the cell but you may find the cell has not executed. This is particularly obvious on print statements when there is no output. If this happens, restart the kernel using the command under the Kernel tab.

# %% [markdown]
# ## Retrieval augmented generation
# In retrieval augmented generation (RAG), an LLM retrieves contextual documents from an external dataset as part of its execution.
# 
# This is useful if we want to ask question about specific documents (e.g., our PDFs, a set of videos, etc).

# %%
#! pip install langchain

# %%
! pip install openai

# %%
! pip install python-dotenv

# %%
import os
import openai
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into the environment, load_dotenv() by default loads from .env in the current directory.

openai.api_key = os.getenv("OPENAI_API_KEY")

# %% [markdown]
# ## PDFs
# Let's load a PDF [transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf) from Andrew Ng's famous CS229 course! These documents are the result of automated transcription so words and sentences are sometimes split unexpectedly.

# %%
# The course will show the pip installs you would need to install packages on your own machine.
# These packages are already installed on this platform and should not be run again.
! pip install pypdf 

# %%
#! pip install -U langchain-community

# %%
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

# %% [markdown]
# Each page is a Document.
# 
# A Document contains text (page_content) and metadata.

# %%
len(pages)

# %%
page = pages[0]

# %%


# %%
type(page)

# %%
type(page.page_content)

# %%
type(page.metadata)

# %%
print(page.page_content[0:200])

# %%
page.metadata

# %% [markdown]
# ## YouTube

# %%
from langchain_community.document_loaders import YoutubeLoader

# %%
#!pip install --upgrade --quiet  youtube-transcript-api

# %%
#!pip install --upgrade --quiet  pytube

# %%
#!pip install --upgrade --quiet yt-dlp

# %%
# Initialize the loader with desired parameters
loader = YoutubeLoader.from_youtube_url(
    #"https://www.youtube.com/watch?v=jGwO_UgTS7I",  # Replace with your video's URL
    "https://www.youtube.com/watch?v=x8FASlLf5ls",
    add_video_info=False,                         # Set to True to fetch video metadata
    language=["zh", "en"],                             # Specify transcript language(s)
    translation="en"                             # Translate transcript if necessary
)

# %%
docs=loader.load()

# %%
docs

# %%
docs[0].metadata

# %%
import yt_dlp

# %%
def fetch_youtube_metadata(url: str):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,  # Don't download the video
        'extract_flat': False,  # Extract full metadata
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title"),
            "description": info.get("description"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "channel": info.get("uploader"),
            "channel_url": info.get("uploader_url"),
            "tags": info.get("tags"),
            "categories": info.get("categories"),
            "thumbnail": info.get("thumbnail"),
            "webpage_url": info.get("webpage_url"),
        }

# %%
url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
#url = "https://www.youtube.com/watch?v=x8FASlLf5ls"
metadata = fetch_youtube_metadata(url)
print(metadata)

# %%
list(metadata.keys())

# %%
docs[0].metadata = metadata.copy()

# %%
docs[0].metadata 

# %%
#!pip install --upgrade langchain langchain-community

# %%
from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)

# %%
#%pip install --upgrade --quiet  yt_dlp
#%pip install --upgrade --quiet  pydub
#%pip install --upgrade --quiet  librosa

# %%
#!pip install --upgrade transformers

# %%
#!pip install torch

# %%
#!pip install ipywidgets --upgrade --quiet

# %%
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# %%
# Two Karpathy lecture videos
urls = ["https://www.youtube.com/watch?v=jGwO_UgTS7I"]
# Directory to save audio files
save_dir = "docs/youtube/"

# %%
# set a flag to switch between local and remote parsing
# change this to True if you want to use local parsing
local = True

# %%
# Transcribe the videos to text
if local:
    loader = GenericLoader(
        YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParserLocal(language="en")
    )
else:
    loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser(language="en"))
docs = loader.load()

# %%
docs[0].page_content

# %%
docs[0].metadata

# %%
metadata| docs[0].metadata

# %%
import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)
from langchain_community.document_loaders.blob_loaders import YoutubeAudioLoader

# Optional: Suppress Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def transcribe_youtube_videos(
    urls: list[str],
    save_dir: str = "docs/youtube",
    local: bool = False,
    language: str = "en"
):
    """
    Downloads and transcribes YouTube videos using OpenAI Whisper.

    Parameters:
        urls (list[str]): List of YouTube video URLs.
        save_dir (str): Directory to save downloaded audio.
        local (bool): Whether to use the local Whisper model.
        language (str): Language code (e.g., "en", "zh") or "auto" for detection.

    Returns:
        list[Document]: LangChain documents containing the transcribed text.
    """
    parser = (
        OpenAIWhisperParserLocal()
        if local
        else OpenAIWhisperParser(language=language)
    )

    loader = GenericLoader(
        YoutubeAudioLoader(urls, save_dir),
        parser
    )
    return loader.load()

# %%
urls = ["https://www.youtube.com/watch?v=x8FASlLf5ls"]
docs = transcribe_youtube_videos(urls, save_dir="docs/youtube/", local=True, language="en")#"en"

print(docs[0].page_content[:300])  

# %%
docs

# %%
docs[1].page_content

# %%
docs[1].metadata

# %%
url = "https://www.youtube.com/watch?v=x8FASlLf5ls"
metadata = fetch_youtube_metadata(url)

# %%
metadata

# %%
docs[1].metadata.update(metadata)

# %%
docs[1].metadata

# %%
docs = load_youtube_document(
    url="https://www.youtube.com/watch?v=x8FASlLf5ls",
    save_dir="docs/youtube/",
    local=True,
    language="zh"
)

# %%
!pip install langdetect

# %%
from langdetect import detect
import re

def detect_lang_from_text(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "zh-cn" or lang == "zh-tw":
            return "zh"
        return lang
    except:
        return "en"  # fallback

# %%
url="https://www.youtube.com/watch?v=bSKX_PPflsk"
# Detect language from title + description
metadata = fetch_youtube_metadata(url)
combined_text = f"{metadata.get('title', '')} {metadata.get('description', '')}"
detected_language = detect_lang_from_text(combined_text)

# %%
detected_language

# %%
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
)
from langchain_community.document_loaders.blob_loaders import YoutubeAudioLoader
import yt_dlp
import os
from langdetect import detect
import re

# Optional: Suppress Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def fetch_youtube_metadata(url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title"),
            "description": info.get("description"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "channel": info.get("uploader"),
            "channel_url": info.get("uploader_url"),
            "tags": info.get("tags"),
            "categories": info.get("categories"),
            "thumbnail": info.get("thumbnail"),
            "webpage_url": info.get("webpage_url"),
        }
    
def detect_lang_from_text(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "zh-cn" or lang == "zh-tw":
            return "zh"
        return lang
    except:
        return "en"  # fallback
    
def load_youtube_document(
    url: str,
    save_dir: str = "docs/youtube",
    local: bool = True,
    language: str = None
) -> List[Document]:
    """
    Loads transcript from YouTube if available, otherwise transcribes from audio.
    Returns enriched LangChain Document(s).
    """
    docs = []

    # Enrich with metadata and detecting language if needed
    metadata = fetch_youtube_metadata(url)
    if language is None: # if no language specified, it will be detected based on meta info
        print("[Language Detection] Language not specified, detecting...")
        combined_text = f"{metadata.get('title', '')} {metadata.get('description', '')}"
        language = detect_lang_from_text(combined_text)

    # Try loading transcript
    try:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            language=[language],
            translation=language
        )
        docs = loader.load()
    except Exception as e:
        print(f"[Transcript Loader] Warning: {e}")

    # If no transcript, fall back to Whisper transcription
    if not docs:
        print("[Fallback] No transcript found. Using Whisper transcription.")
        parser = (
            OpenAIWhisperParserLocal()
            if local
            else OpenAIWhisperParser(language=language)
        )
        audio_loader = GenericLoader(YoutubeAudioLoader([url], save_dir), parser)
        docs = audio_loader.load()


    for doc in docs:
        doc.metadata.update(metadata)

    return docs

# %%
docs=load_youtube_document(
      url="https://www.youtube.com/watch?v=tkFDeadKz2I",
      save_dir="docs/youtube",
      local=True,
      language=None) 

# %%
docs

# %%
docs[1].metadata

# %% [markdown]
# **Note**: This can take several minutes to complete.

# %% [markdown]
# ## URLs

# %%
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")

# %%
docs = loader.load()

# %%
print(docs[0].page_content[:500])

# %% [markdown]
# ## Notion

# %% [markdown]
# Follow steps [here](https://python.langchain.com/docs/modules/data_connection/document_loaders/integrations/notion) for an example Notion site such as [this one](https://yolospace.notion.site/Blendle-s-Employee-Handbook-e31bff7da17346ee99f531087d8b133f):
# 
# - Duplicate the page into your own Notion space and export as Markdown / CSV.
# - Unzip it and save it as a folder that contains the markdown file for the Notion page.

# %%
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()

# %%
print(docs[0].page_content[0:200])

# %%
docs[0].metadata


