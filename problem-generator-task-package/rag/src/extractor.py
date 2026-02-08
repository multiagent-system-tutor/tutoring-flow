import os
import re
import json
import pdfplumber
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
from .utils import setup_logger, time_tracker

logger = setup_logger("extractor")

class PDFExtractor:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
        return text

    def parse_questions(self, text: str, source_file: str) -> List[Dict]:
        questions = []
        
        p1 = r'\(\s*Q\s*(\d+)\s*[-–]\s*([^)]+?)\s*(?:\d+\s*Points?)?\)'

        p2 = r'\[\s*Question\s*(\d+)\s*[-–]\s*([^\]]+?)(?:\s*:\s*\d+\s*Points?)?\s*\]'
        
        combined_pattern = f"({p1})|({p2})"
        
        matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))
        
        parsed_matches = []
        for m in matches:
            if m.group(1): 
                parsed_matches.append({
                    'start': m.start(),
                    'end': m.end(),
                    'full': m.group(0),
                    'num': m.group(2),
                    'topic': m.group(3).strip()
                })
            elif m.group(4): 
                parsed_matches.append({
                    'start': m.start(),
                    'end': m.end(),
                    'full': m.group(0),
                    'num': m.group(5),
                    'topic': m.group(6).strip()
                })

        for i, meta in enumerate(parsed_matches):
            start_idx = meta['start']
            end_idx = parsed_matches[i+1]['start'] if i + 1 < len(parsed_matches) else len(text)
            
            header = meta['full']
            q_num = meta['num']
            topic = meta['topic']
            
            full_content = text[start_idx:end_idx]
            
            content_body = full_content.replace(header, "", 1).strip()
         
            split_markers = [
                "Tuliskan jawaban di sini (Write your answer at the given box)!",
                "Tuliskan jawaban di sini",
                "Write your answer at the given box"
            ]
            
            question_text = content_body
            answer_text = ""
            
            for marker in split_markers:
                if marker in content_body:
                    parts = content_body.split(marker)
                    question_text = parts[0].strip()
                    if len(parts) > 1:
                        answer_text = parts[1].strip()
                    break
            
            clo = self.find_clo_context(text, start_idx)

            entry = {
                "source_file": source_file,
                "question_id": f"{os.path.basename(source_file)}_Q{q_num}",
                "topic": topic,
                "clo": clo,
                "question": question_text,
                "answer": answer_text,
                "full_text": full_content
            }
            questions.append(entry)
            
        return questions

    def find_clo_context(self, full_text: str, current_idx: int) -> str:
        search_window = full_text[max(0, current_idx-1000):current_idx]
        clo_match = re.search(r'(CLO-\d+-\d+)', search_window)
        if clo_match:
            return clo_match.group(1)
        
        return "Unknown"

    @time_tracker
    def run(self):
        pdf_files = [f for f in os.listdir(self.raw_dir) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files in {self.raw_dir}")
        
        all_data = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            full_path = os.path.join(self.raw_dir, pdf_file)
            logger.info(f"Processing {pdf_file}...")
            
            text = self.extract_text_from_pdf(full_path)
            questions = self.parse_questions(text, pdf_file)
            
            if questions:
                logger.info(f"Extracted {len(questions)} questions from {pdf_file}")
                all_data.extend(questions)
            else:
                logger.warning(f"No questions found in {pdf_file} matching pattern.")

        output_file = os.path.join(self.processed_dir, "dataset.jsonl")
        logger.info(f"Saving {len(all_data)} entries to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in all_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "dataset", "raw")
    processed_dir = os.path.join(base_dir, "dataset", "processed")
    
    extractor = PDFExtractor(
        raw_dir=raw_dir,
        processed_dir=processed_dir
    )
    extractor.run()
