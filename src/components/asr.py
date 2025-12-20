from ..utils.timer import Timer

class ASRComponent:
    def __init__(self, config=None):
        self.config = config or {}

    def run(self, audio):
        """
        Args:
            audio: Path to audio file or audio data.
        Returns:
            dict: {'time': float, 'text': str, 'voice': ...}
        """
        result = {}
        with Timer() as t:
            # TODO: Implement ASR logic (speech to text)
            pass
        
        result['time'] = t.interval
        result['text'] = ""
        return result
