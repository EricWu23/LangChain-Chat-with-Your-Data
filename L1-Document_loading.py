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
! pip install langchain

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
! pip install -U langchain-community

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
!pip install --upgrade --quiet  youtube-transcript-api

# %%
!pip install --upgrade --quiet  pytube

# %%
!pip install --upgrade --quiet yt-dlp

# %%
# Initialize the loader with desired parameters
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=jGwO_UgTS7I",  # Replace with your video's URL
    add_video_info=False,                         # Set to True to fetch video metadata
    language=["en"],                             # Specify transcript language(s)
    translation="en"                             # Translate transcript if necessary
)

# %%
docs=loader.load()

# %%
docs[0].page_content

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
metadata = fetch_youtube_metadata(url)
print(metadata)

# %%
list(metadata.keys())

# %%
docs[0].metadata = metadata.copy()

# %%
docs[0].metadata 

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


