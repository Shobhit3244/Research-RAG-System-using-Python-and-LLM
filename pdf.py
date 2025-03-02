import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("Building Index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))

    return index


pdf_data = [PDFReader().load_data(file=os.path.join("data", x)) for x in os.listdir("data") if x.endswith(".pdf")]
pdf_index = [get_index(x, y) for x, y in zip(pdf_data, os.listdir("data")) if y.endswith(".pdf")]
QEngines = [QueryEngineTool(
    query_engine=x.as_query_engine(),
    metadata=ToolMetadata(
        name=y,
        description=f"this is the research paper on {y} and gives information about it."
    )
) for x, y in zip(pdf_index, os.listdir("data")) if y.endswith(".pdf")]
