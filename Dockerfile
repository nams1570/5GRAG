FROM python:3.11.11
WORKDIR /

ENV PIP_DEFAULT_TIMEOUT=600
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./CollectionNames.py ./CollectionNames.py
COPY ./MetadataAwareChunker.py ./MetadataAwareChunker.py
COPY ./MultiStageRetriever.py ./MultiStageRetriever.py
COPY ./RAGQAEngine.py ./RAGQAEngine.py
COPY ./controller.py ./controller.py
COPY ./ds_server.py ./ds_server.py
COPY ./settings.py ./settings.py
COPY ./utils.py ./utils.py
COPY ./prompt.txt ./prompt.txt
COPY ./ReferenceExtractor.py ./ReferenceExtractor.py
COPY ./HypotheticalDocGenerator.py ./HypotheticalDocGenerator.py
COPY ./DBClient.py ./DBClient.py
COPY ./ChangeTracker.py ./ChangeTracker.py

COPY ./settings.yml ./settings.yml


CMD ["fastapi", "run", "ds_server.py", "--port", "80"]
