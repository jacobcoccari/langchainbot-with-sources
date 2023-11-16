

def format_source_string(documents):
    unique_source_documents = set(
        [
            "[" + source_document.metadata["source"] + "](" + source_document.metadata["source"] + ")"
            for source_document in documents
        ]
    )
    source_string = ""
    for source_document in unique_source_documents:
        source_string = (
            source_string
            + source_document
            + """
            """
        )
    return source_string