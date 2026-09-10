"""Document processing module for loading and splitting documents."""

import os
import re
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import requests
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    @staticmethod
    def _normalize_source(source: Union[str, Path]) -> str:
        """Normalize values copied from terminals, chat, or rendered Markdown."""
        value = str(source).replace("\u00a0", " ").strip()
        markdown_link = re.fullmatch(r"\[[^]]*\]\((https?://[^)]+)\)", value)
        if markdown_link:
            value = markdown_link.group(1).strip()
        return value

    @staticmethod
    def _is_url(source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _github_username(source: str) -> Union[str, None]:
        """Return the username when source is a GitHub profile URL."""
        parsed = urlparse(source)
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            return None

        path_parts = [part for part in parsed.path.split("/") if part]
        return path_parts[0] if len(path_parts) == 1 else None

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_from_github_user(self, username: str) -> List[Document]:
        """Load public repository metadata and READMEs for a GitHub user."""
        token = os.getenv("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        repositories = []
        page = 1
        while True:
            response = requests.get(
                f"https://api.github.com/users/{username}/repos",
                headers=headers,
                params={
                    "type": "owner",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": 100,
                    "page": page,
                },
                timeout=30,
            )
            response.raise_for_status()
            page_repositories = response.json()
            repositories.extend(page_repositories)

            if len(page_repositories) < 100:
                break
            page += 1

        documents: List[Document] = []
        for repository in repositories:
            if repository.get("fork"):
                continue

            readme_response = requests.get(
                f"https://api.github.com/repos/{username}/{repository['name']}/readme",
                headers={
                    **headers,
                    "Accept": "application/vnd.github.raw+json",
                },
                timeout=30,
            )
            if readme_response.status_code == 200:
                readme = readme_response.text
            elif readme_response.status_code == 404:
                readme = "No README available."
            else:
                readme_response.raise_for_status()
                readme = "No README available."

            topics = ", ".join(repository.get("topics") or []) or "None"
            content = "\n".join(
                [
                    f"Project: {repository['name']}",
                    f"Description: {repository.get('description') or 'Not provided'}",
                    f"Primary language: {repository.get('language') or 'Not specified'}",
                    f"Topics: {topics}",
                    f"Repository: {repository['html_url']}",
                    f"Homepage: {repository.get('homepage') or 'None'}",
                    "",
                    "README:",
                    readme,
                ]
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": repository["html_url"],
                        "repository": repository["name"],
                        "type": "github_repository",
                    },
                )
            )

        return documents

    def load_documents(self, sources: List[Union[str, Path]]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files

        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for source in sources:
            src = self._normalize_source(source)

            github_username = self._github_username(src)
            if github_username:
                docs.extend(self.load_from_github_user(github_username))
                continue

            if self._is_url(src):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src).expanduser()
            if path.is_dir():  # PDF directory
                docs.extend(self.load_from_pdf_dir(path))
            elif path.is_file() and path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            elif path.is_file() and path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {source}. "
                    "Use an HTTP(S) URL, a .txt/.pdf file, or a PDF directory."
                )
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)

    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents

        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        print(self.split_documents(docs))
        return self.split_documents(docs)
