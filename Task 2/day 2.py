import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_documents():
    # Exercise 1: Load PDFs
    loaders = [
        PyPDFLoader("tvk_report.pdf"),
        PyPDFLoader("dmk_report.pdf")
    ]
    
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    # Exercise 2: Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    
    # Exercise 3: Attach Metadata
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    
    for chunk in chunks:
        # Extract existing info from loader (like page number and source)
        source_path = chunk.metadata.get("source", "unknown")
        page_idx = chunk.metadata.get("page", 0)
        
        # Determine source type based on filename
        source_type = "research_paper" if "report" in source_path else "notes"
        
        # Update metadata dictionary
        chunk.metadata.update({
            "filename": source_path.split("/")[-1],
            "page_number": page_idx + 1,  # Convert 0-indexed to 1-indexed
            "upload_date": current_date,
            "source_type": source_type
        })
        
    return chunks

# Exercise 4: Build a Filter Function
def filter_chunks(chunks, **filters):
    """
    Returns only the chunks matching the given metadata key-value pairs.
    """
    filtered_list = []
    for chunk in chunks:
        match = True
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                match = False
                break
        if match:
            filtered_list.append(chunk)
    return filtered_list

# Exercise 5: Test Your Filters
if __name__ == "__main__":
    print("--- Processing Documents ---")
    all_chunks = process_documents()
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Test 1: Filter by filename
    tvk_chunks = filter_chunks(all_chunks, filename="tvk_report.pdf")
    print(f"Chunks for tvk_report.pdf: {len(tvk_chunks)}")
    
    # Test 2: Filter by specific page and file
    specific_page = filter_chunks(all_chunks, filename="dmk_report.pdf", page_number=5)
    print(f"Chunks on Page 5 of dmk_report.pdf: {len(specific_page)}")
    if specific_page:
        print(f"Sample Content: {specific_page[0].page_content[:100]}...")

    # Test 3: Filter by source type
    paper_chunks = filter_chunks(all_chunks, source_type="research_paper")
    print(f"Total Research Paper chunks: {len(paper_chunks)}")