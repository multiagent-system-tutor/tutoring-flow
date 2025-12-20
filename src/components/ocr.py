from ..utils.timer import Timer

class OCRComponent:
    def __init__(self, config=None):
        self.config = config or {}

    def run(self, image):
        """
        Args:
            image: Path to image or image data.
        Returns:
            dict: {'time': float, 'text': str, 'image': ...}
        """
        result = {}
        with Timer() as t:
            # TODO: Implement OCR logic (extract text from clean/dirty image)
            pass
        
        result['time'] = t.interval
        result['text'] = ""
        # result['image'] = ...
        return result
