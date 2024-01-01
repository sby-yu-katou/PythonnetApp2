import os
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from trafilatura import fetch_url, extract
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

import pandas as pd

# コンストラクタでデータベースの初期化をおこなう
# 既に作成済みの場合はロードするだけ
# 未作成の場合は初期化して作成する

# データベースの初期化では適当な URL からテキストだけを抽出して保存する

# 質問文を受け付けて LLM に流す責務は別のクラスでおこなう

class DbOperator():
    def __init__(self):
        self.__persistent_directory = "./chromadb"
        # self.__embeddings = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-large")
        if not os.path.exists(self.__persistent_directory):
            # なければ作る
            self.__init()
        else:
            self.__db = chromadb.PersistentClient(path = self.__persistent_directory)
            # self.client = Chroma(self.persistent_directory, embeddings = self.__embeddings)
        self.__chunk = self.__db.get_collection("chunk")

    def check(self):
        if self.__chunk:
            print(self.__chunk.count())
            return True
        else:
            return False

    def __init(self):
        url = "https://ja.wikipedia.org/wiki/%E3%83%89%E3%83%A9%E3%82%B4%E3%83%B3%E3%83%9C%E3%83%BC%E3%83%AB"
        document = fetch_url(url)
        text = extract(document)
        # テキストをチャンクに分割
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator = "\n",
            chunk_size = 300,
            chunk_overlap = 20,
        )
        texts = text_splitter.create_documents(text_splitter.split_text(text))
        df = pd.DataFrame(texts, columns = [ "page_content", "metadata", "type" ])

        cache_dir = "./.cache/models/intfloat_multilingual-e5-large/"
        # SentenceTransformer モデルを読み込む
        model = SentenceTransformer("intfloat/multilingual-e5-large", cache_folder = cache_dir)

        # 埋め込みを生成するための関数
        def generate_embeddings(text):
            embeddings = model.encode(text)
            return embeddings.tolist()

        # print(df["metadata"])
        # print("########################")
        # print(df["page_content"])
        # print("########################")
        # print(df["type"])

        # df["id"] に "Id00000001" というようなユニークな ID を振る
        df["id"] = df.index.map(lambda x: "Id" + str(x).zfill(8))
        print(df["id"])

        df["embeddings"] = df["page_content"].apply(generate_embeddings)

        self.__db = chromadb.PersistentClient(path = self.__persistent_directory)
        self.__chunk = self.__db.create_collection("chunk")
        self.__chunk.add(
            documents = df["page_content"].tolist(),
            embeddings = df["embeddings"].tolist(),
            ids = df["id"].tolist()
        )
