import os
import re
import sys
import pickle
import hashlib
import PyPDF2
import Levenshtein
from tqdm import tqdm
from collections import defaultdict

def get_cache_path(directory_path):
    """Create and return path to cache directory"""
    cache_dir = os.path.join(directory_path, ".pdf_search_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def get_file_hash(file_path):
    """Generate a hash for the file path and modification time"""
    file_stat = os.stat(file_path)
    file_info = f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
    return hashlib.md5(file_info.encode()).hexdigest()

def cache_pdf_text(pdf_path, cache_dir):
    """Extract text from PDF and cache it, or return cached version if available"""
    file_hash = get_file_hash(pdf_path)
    cache_file = os.path.join(cache_dir, f"{file_hash}.pkl")
    
    # If cache exists and is valid, load it
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Помилка кешу для {pdf_path}: {e}")
    
    # Otherwise extract text and cache it
    text_by_page = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                try:
                    text = page.extract_text()
                    text_by_page.append(text if text else "")
                except Exception as e:
                    text_by_page.append("")
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(text_by_page, f)
            
            return text_by_page
    except Exception as e:
        print(f"Помилка обробки {pdf_path}: {e}")
        return []

def fuzzy_search(text, search_term, max_distance):
    """Find matches using Levenshtein distance for fuzzy matching"""
    results = []
    words = re.findall(r'\b\w+\b', text, re.UNICODE)
    
    # For multi-word search terms
    if ' ' in search_term:
        search_words = search_term.split()
        n = len(search_words)
        
        for i in range(len(words) - n + 1):
            total_distance = 0
            match = True
            
            for j in range(n):
                if i+j < len(words):
                    word_distance = Levenshtein.distance(words[i+j].lower(), search_words[j].lower())
                    word_max_dist = max(1, len(search_words[j]) // 3)
                    
                    if word_distance > word_max_dist:
                        match = False
                        break
                        
                    total_distance += word_distance
            
            if match and total_distance <= max_distance:
                results.append({'distance': total_distance})
    else:
        # Single word search
        for word in words:
            distance = Levenshtein.distance(word.lower(), search_term.lower())
            max_acceptable = max(1, len(search_term) // 3)
            
            if distance <= max_acceptable and distance <= max_distance:
                results.append({'distance': distance})
    
    return results

def search_pdfs(directory_path, search_term, max_distance=2):
    """
    Search for a term in all PDF files and return simplified match information.
    
    Args:
        directory_path: Path to directory containing PDF files
        search_term: Term to search for
        max_distance: Maximum Levenshtein distance for matches
    """
    file_matches = defaultdict(lambda: {'count': 0, 'pages': set()})
    cache_dir = get_cache_path(directory_path)
    
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"PDF-файлів не знайдено в {directory_path}")
        return {}
    
    print(f"Пошук '{search_term}' у {len(pdf_files)} PDF-файлах...")
    
    # Process each PDF file
    for pdf_file in tqdm(pdf_files):
        file_path = os.path.join(directory_path, pdf_file)
        
        # Get cached text or extract if not cached
        pages_text = cache_pdf_text(file_path, cache_dir)
        
        # Search in each page
        for page_num, text in enumerate(pages_text):
            if not text:
                continue
                
            # Find fuzzy matches in page
            matches = fuzzy_search(text, search_term, max_distance)
            
            # If matches found, update file match information
            if matches:
                file_matches[pdf_file]['count'] += len(matches)
                file_matches[pdf_file]['pages'].add(page_num + 1)
    
    return file_matches

def main():
    if len(sys.argv) < 3:
        print("Використання: python pdf_search.py <шлях_до_директорії> <пошуковий_термін> [макс_відстань]")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    search_term = sys.argv[2]
    max_distance = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    if not os.path.isdir(directory_path):
        print(f"Помилка: {directory_path} не є дійсною директорією")
        sys.exit(1)
    
    file_matches = search_pdfs(directory_path, search_term, max_distance)
    
    if not file_matches:
        print("Збігів не знайдено.")
        return
    
    print(f"\nРезультати пошуку для '{search_term}':")
    print("-" * 80)
    
    # Sort files by match count (descending)
    sorted_files = sorted(file_matches.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Print results
    for i, (filename, match_info) in enumerate(sorted_files, 1):
        # Sort page numbers
        pages = sorted(match_info['pages'])
        
        # Format page ranges (e.g., 1-3, 5, 7-9 instead of 1,2,3,5,7,8,9)
        page_ranges = []
        start = end = pages[0]
        
        for page in pages[1:]:
            if page == end + 1:
                end = page
            else:
                if start == end:
                    page_ranges.append(str(start))
                else:
                    page_ranges.append(f"{start}-{end}")
                start = end = page
        
        # Add the last range
        if start == end:
            page_ranges.append(str(start))
        else:
            page_ranges.append(f"{start}-{end}")
        
        pages_str = ", ".join(page_ranges)
        
        print(f"{i}. {filename}")
        print(f"   Кількість збігів: {match_info['count']}")
        print(f"   Сторінки: {pages_str}")
        print()

if __name__ == "__main__":
    main()