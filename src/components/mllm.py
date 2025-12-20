from ..utils.timer import Timer

class MLLMComponent:
    def __init__(self, config=None):
        self.config = config or {}

    def run(self, image):
        """
        Args:
            image: Path to image or image data.
        Returns:
            dict: {'time': float, 'text': str, ...}
        """
        result = {}
        with Timer() as t:
            # TODO: Implement MLLM logic
            pass
        
        result['time'] = t.interval
        result['text'] = ""
        return result
