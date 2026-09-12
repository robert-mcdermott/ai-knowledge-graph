"""
Text processing utilities for the knowledge graph generator.
"""

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split a text into chunks of words with overlap.

    Args:
        text: The input text to chunk
        chunk_size: The size of each chunk in words
        overlap: The number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    # Split text into words
    words = text.split()

    # If text is smaller than chunk size, return it as a single chunk
    if len(words) <= chunk_size:
        return [text]

    # Create chunks with overlap
    chunks = []
    start = 0

    while start < len(words):
        # Calculate end position for this chunk
        end = min(start + chunk_size, len(words))

        # Join words for this chunk
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        # Move start position for next chunk, accounting for overlap
        start = end - overlap

        # If we're near the end and the last chunk would be too small, just exit
        if start < len(words) and start + chunk_size - overlap >= len(words):
            # Add remaining words as the final chunk
            final_chunk = ' '.join(words[start:])
            chunks.append(final_chunk)
            break

    return chunks
