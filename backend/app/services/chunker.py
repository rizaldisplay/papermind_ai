import re

class ChunkerService:
    @staticmethod
    def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
        """
        Simulasi chunking dokumen dengan ekstraksi metadata halaman dan bab.
        Dalam realita, Anda bisa menggunakan PyPDF2 atau LangChain PDF Loader.
        """
        # Simulasi deteksi Bab dan Halaman menggunakan regex sederhana
        pages = text.split("[PAGE_BREAK]")
        chunks_output = []
        
        current_chapter = "Introduction"
        
        for page_idx, page_content in enumerate(pages):
            page_num = page_idx + 1
            
            # Deteksi jika ada penanda bab baru
            chap_match = re.search(r'(Chapter\s+\d+|Bab\s+\d+|Introduction|Conclusion)', page_content, re.IGNORECASE)
            if chap_match:
                current_chapter = chap_match.group(0)
                
            words = page_content.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                if len(chunk_text.strip()) > 10:
                    chunks_output.append({
                        "content": chunk_text,
                        "page_number": page_num,
                        "chapter": current_chapter
                    })
        return chunks_output