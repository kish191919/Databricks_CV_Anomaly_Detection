import base64
from llm import Label
from pathlib import Path

class GenerateLabel(Label):
    def __init__(self, context_prompt_path):
        super().__init__(context_prompt_path)
    
    def encode_image(self, image_path:Path):
        """Encode an image to a base64 string"""
        try:
            with image_path.open('rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

        except Exception as e:
            print(f"Error Encoding image : {image_path}, {e}")
            return None

if __name__  == "__main__":
    contextpromptpath = Path('prompt') / 'label.md'
    generatelabel = GenerateLabel(context_prompt_path=contextpromptpath)
    
    image_path = Path('frames') / 'frame_0000.jpg'

    b64_image = generatelabel.encode_image(image_path)
    response = generatelabel.run(b64_image)
    print(response)



        