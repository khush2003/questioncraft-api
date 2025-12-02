from langchain_core.documents import Document
import re

def create_chunks_with_overlap(text, chunk_size, overlap_size, page_number):
    chunks = []
    position = 0

    while position < len(text):
        chunk = text[position:position + chunk_size]
        metadata = {
            'chunk_index': len(chunks),
            'start_position': position,
            'end_position': position + len(chunk),
            'chunk_length': len(chunk),
            'page_number': page_number
        }

        chunks.append({'chunk': chunk, 'metadata': metadata})

        # Move position forward with overlap
        position += chunk_size - overlap_size

    return chunks

def save_documents_to_file(documents, filename="documents.txt"):
                with open(filename, "w", encoding="utf-8") as file:
                    for doc in documents:
                        file.write(f"Page Content: $$( {doc.page_content} )$$  \nMetadata: $$( {doc.metadata} )$$ \n\n")
                        
def load_documents_from_file(filename="documents.txt"):
    documents = []
    with open(filename, "r",  encoding="utf-8") as file:
        content = file.read().strip()
        pattern = re.compile(r"Page Content: \$\$\((.*?)\)\$\$  \nMetadata: \$\$\((.*?)\)\$\$", re.DOTALL)
        matches = pattern.findall(content)
        for match in matches:
            page_content, metadata = match
            metadata = eval(metadata.strip())
            documents.append(Document(page_content=page_content.strip(), metadata=metadata))
    return documents


def convert_to_list(input_string: str) -> list:
    # Remove the square brackets and split by comma
    items = input_string.strip('[]').split(',')
    # Strip any extra whitespace and return the list
    return [item.strip() for item in items]

__all__ = ['create_chunks_with_overlap']