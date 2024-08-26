import os
import requests
import subprocess
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Helpers:

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 102):
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.links: list = []
        self.agg_chunks: list = []

    def get_links(self, base, url, lvl=3) -> None:
        """link scraper"""
        if lvl == 0:
            return

        self.links.append(url)
        loader = AsyncChromiumLoader([url])

        try:
            response = loader.load()

            if not response[0]:
                return

            soup = BeautifulSoup(response[0].page_content, "html.parser")

            for link in soup.find_all("a"):
                href = link.get("href")
                if href:
                    full_url = urljoin(url, href)
                    if base in full_url:
                        if full_url not in self.links:
                            self.get_links(base, full_url, lvl - 1)
        except Exception as e:
            os.write(1, f"get links exception: {e}".encode())
            return

    def get_data(self) -> list:
        """chunk and return data"""
        os.write(1, f"links: {self.links}".encode())
        if self.links:
            loader = AsyncChromiumLoader(self.links)
            try:
                docs = loader.load()
            except:
                subprocess.run(["playwright", "install"], check=False)
                docs = loader.load()
            transformed_docs = BeautifulSoupTransformer().transform_documents(
                docs, tags_to_extract=["p", "div", "li"]
            )

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

            chunks = splitter.split_documents(transformed_docs)
            self.agg_chunks += chunks
        else:
            self.agg_chunks += chunks
