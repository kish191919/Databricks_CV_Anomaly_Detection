import base64
from llm import Label
from pathlib import Path
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import OrderedDict



class GenerateLabel(Label):
    def __init__(self,input_dir:Path, output_dir:Path, context_prompt_path):
        super().__init__(context_prompt_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._validation_path()

    def _validation_path(self):
        if not all([isinstance(self.input_dir, Path), isinstance(self.output_dir, Path)]):
            raise ValueError(f'input_dir and output_dir must be a pathlib.Path')
    
    def encode_image(self, image_path:Path):
        """Encode an image to a base64 string"""
        try:
            with image_path.open('rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

        except Exception as e:
            print(f"Error Encoding image : {image_path}, {e}")
            return None
    
    def process_images(self, image_path) -> dict:
        """Process an image by encoding it into base64 and get an label from LLM"""

        base64_image = self.encode_image(image_path)
        if not base64_image:
            raise ValueError(f"Fail to encode image : {image_path}")
        json_response = self.run(base64_image)
        return self.parse_json_str(json_response)
    

    def exe_label(self, max_workers:int=4):
        files = list(self.input_dir.glob("*.jpg"))
        if not files:
            print(f"No images found in : {self.input_dir}")
            return None
        
        results = OrderedDict()
        with ThreadPoolExecutor(max_workers=max_workers) as executors:
            futures = {executors.submit(self.process_images, file) : file for file in files}
            # futures = [executors.submit(self.process_images, file) for file in files]

            for future in tqdm(as_completed(futures), total=len(files), desc="Processing..."):
                file = futures[future]
                try:
                    res = future.result()
                    results[file.name] = res 
                except Exception as e:
                    print(f"Failed: {file}, {e}")
        
        # Save
        self.output_dir.mkdir(exist_ok=True)
        output_file_path = self.output_dir / "label.json"

        with output_file_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4 )
            
    
    def parse_json_str(self, json_str:str) -> dict:
        """Extract JSON from a string, handling edge cases"""

        try:
            match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
            if match:
                json_data = match.group(1).strip()
            else:
                json_data = json_str.strip()
            return json.loads(json_data)

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error Parsing JSON: {e} ")
            return None
        

if __name__  == "__main__":
    contextpromptpath = Path('prompt') / 'label.md'    
    input_dir = Path('frames')
    output_dir = Path('labels')

    generatelabel = GenerateLabel(input_dir=input_dir, output_dir=output_dir, context_prompt_path=contextpromptpath)
    generatelabel.exe_label(max_workers=5)

    # image_path = Path('frames') / 'frame_0000.jpg'
    # b64_image = generatelabel.encode_image(image_path)
    # response = generatelabel.run(b64_image)
    # label = generatelabel.parse_json_str(response)
    # print(f"{type(response)}, {response}")
    # print(f"{type(label)}, {label}")



        