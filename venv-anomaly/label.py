import base64
from llm import Label
from pathlib import Path
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import datetime
import pytz


class GenerateLabel(Label):

    def __init__(self,input_dir:Path, output_dir:Path, context_prompt_path):
        super().__init__(context_prompt_path)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir_temp = self.output_dir / 'temp'
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
    

    def process_images(self, image_path) -> None:
        """Process an image by encoding it into base64 and get an label from LLM"""

        print("process_images")

        print(f"image_path: {image_path}")

        base64_image = self.encode_image(image_path)
        if not base64_image:
            raise ValueError(f"Fail to encode image : {image_path}")
        
        json_response = self.run(base64_image)

        # [Cache] Save the JSON response to a file
        response = self.parse_json_str(json_response)
        self.output_dir_temp.mkdir(exist_ok=True, parents=True)
        output_path = self.output_dir_temp / f'{image_path.stem}.json'
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=4)
        
        # 추가
        return response

    

    def label(self, max_workers:int=4):
        """Generates labels for images, using threading for parallel execution. """
        files = list(self.input_dir.glob("*.jpg"))
        if not files:
            print(f"No images found in : {self.input_dir}")
            return None
        
        frames = self.load_labels()
        if frames:
            files = sorted([file for file in files if file.stem not in frames])

            if not files:
                print(f'No new labels to process')
                return None

        
        results = OrderedDict()
        with ThreadPoolExecutor(max_workers=max_workers) as executors:
            futures = {executors.submit(self.process_images, file) : file for file in files}
            # futures = [executors.submit(self.process_images, file) for file in files]

            for future in tqdm(as_completed(futures), total=len(files), desc="Processing..."):
                file = futures[future]
                try:
                    res = future.result()
                    results[file.stem] = res 
                except Exception as e:
                    print(f"Failed: {file}, {e}")

        # [Cache] Load the JSON responses from the files
        collected = self.gather_label_results()
        final = collected if collected else results

        # 키 기준 정렬 (frame_0000, frame_0001, ...)
        # key=lambda kv: int(re.search(r'\d+', kv[0]).group(0))
        final_sorted = OrderedDict(sorted(final.items(), key=lambda kv: kv[0]))
    
        # Save
        self.output_dir.mkdir(exist_ok=True)
        output_file_path = self.output_dir / "label.json"

        with output_file_path.open('w', encoding='utf-8') as f:
             json.dump(final_sorted, f, ensure_ascii=False, indent=4 )

    def gather_label_results(self):
        """"Saves the results to a JSON file"""
        label_files = sorted(self.output_dir_temp.glob('*.json'))

        # Use an OrderedDict to maintain the order of the files
        results = OrderedDict()

        for file in label_files:
            try:
                with file.open('r', encoding='utf-8') as f:
                    data = json.load(f, object_hook=OrderedDict)
                    results[file.stem] = data
            except Exception as e:
                print(f"Error loading JSON files: {e}")
            else:
                file.unlink() # Delete the file after loading
        
        self.output_dir.mkdir(exist_ok=True)
        tz = pytz.timezone('US/Eastern')
        now = datetime.datetime.now(tz)
        timestamp = now.isoformat('T', 'seconds').replace(':','-')

        output_file_path = self.output_dir / f'labels_{timestamp}.json'
        with output_file_path.open('w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        self.output_dir_temp.rmdir()

        return results
    
    def load_labels(self) ->set:
        """Loads the labels from a JSON file"""
        label_files = sorted(self.output_dir.glob('labels_*.json'))
        if not label_files:
            return None
        frames = set()
        for file in label_files:
            with file.open('r', encoding='utf-8') as f:
                labels = json.load(f, object_pairs_hook=OrderedDict)
                frames.update(labels.keys())
        return frames
       
# def label_df(file_path:Path) -> pd.DataFrame:
#     with file_path.open('r', encoding='utf-8') as f:
#         data = json.load(f)
#     # return pd.DataFrame.from_dict(data, orient="index").reset_index() # key -> index
#     return pd.DataFrame.from_dict(data, orient="index").sort_index().reset_index() # key -> index

def labeltodataframe(label_path:Path) -> pd.DataFrame:
    with label_path.open('r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    df = pd.DataFrame.from_dict(label_data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index':'image_name', 'animal':'label'}, inplace=True)
    return df


if __name__  == "__main__":
    contextpromptpath = Path('prompt') / 'label.md'    
    input_dir = Path('frames')
    output_dir = Path('labels')

    generatelabel = GenerateLabel(input_dir=input_dir, output_dir=output_dir, context_prompt_path=contextpromptpath)
    generatelabel.label(max_workers=5)

    label_data_path = Path('labels') / 'label.json'
    df = labeltodataframe(label_data_path)
    # df = label_df(label_data_path)
    
    print(df)
    df.to_csv(output_dir / 'labels.csv', index=False)

    # image_path = Path('frames') / 'frame_0000.jpg'
    # b64_image = generatelabel.encode_image(image_path)
    # response = generatelabel.run(b64_image)
    # label = generatelabel.parse_json_str(response)
    # print(f"{type(response)}, {response}")
    # print(f"{type(label)}, {label}")




        