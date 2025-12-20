from ..utils.timer import Timer

class ProblemRAGComponent:
    def __init__(self, config=None):
        self.config = config or {}

    def run(self, plan, todays_date):
        """
        Args:
            plan: str
            todays_date: str
        Returns:
            dict: {'time': float, 'context': str}
        """
        result = {}
        with Timer() as t:
            # TODO: Implement RAG for Problem Generator (Bank Soal)
            pass
        
        result['time'] = t.interval
        result['context'] = ""
        return result
