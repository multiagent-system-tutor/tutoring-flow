import json
from typing import Dict, List, Optional

class SyllabusMapper:
    
    DEFAULT_MAPPING = {
        "1": "Sequential, Input/Output, Variables & Data Types",
        "2": "Sequential, Basic Arithmetic Operations",
        "3": "Sequential, Basic Logic (Pre-Conditionals)",
        "4": "Looping (For Loops)",
        "5": "Looping (While/Do-While Loops)",
        "6": "Arrays (1D)",
        "7": "Functions/Methods",
        # Add more as needed
    }

    def __init__(self, mapping_file: Optional[str] = None):
        self.mapping = self.DEFAULT_MAPPING.copy()
        if mapping_file:
            self._load_mapping(mapping_file)

    def _load_mapping(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                custom_map = json.load(f)
                self.mapping.update(custom_map)
        except FileNotFoundError:
            print(f"Warning: Mapping file {filepath} not found. Using defaults.")

    def get_topic_by_week(self, week: str) -> str:
        week_sanitized = week.lower().replace("minggu", "").strip()

        if week_sanitized in self.mapping:
            return self.mapping[week_sanitized]
        
        return "General Programming Topic"

    def get_all_topics(self) -> List[str]:
        return list(self.mapping.values())

if __name__ == "__main__":
    mapper = SyllabusMapper()
    print(f"Week 1 Topic: {mapper.get_topic_by_week('1')}")
    print(f"Week 2 Topic: {mapper.get_topic_by_week('2')}")
    print(f"Week 3 Topic: {mapper.get_topic_by_week('3')}")
    print(f"Week 4 Topic: {mapper.get_topic_by_week('Minggu 4')}")
