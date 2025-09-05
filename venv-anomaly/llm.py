from dotenv import load_dotenv
from openai import AzureOpenAI
import os
from pathlib import Path

load_dotenv()

class AOAIConfig:
    api_key = os.getenv("AOAI_API_KEY")
    api_version = os.getenv("AOAI_API_VERSION")
    api_endpoint = os.getenv("AOAI_API_ENDPOINT")

    def __init(self):
        if not all([self.api_key, self.api_version, self.api_endpoint]):
            raise ValueError("Missing required AOAI configuration in environment variables.")

class Label(AOAIConfig):
    def __init__(self, context_prompt_path:Path, llm_deployed_name:str='o4-mini-team1'):
        super().__init__()
    
        self.llm_model = llm_deployed_name
        self.client = self._client()
        self.context_prompt_path = context_prompt_path
        self.context_prompt = self._get_context_prompt()
        
    
    def _get_context_prompt(self) -> str:
        if not isinstance(self.context_prompt_path, Path):
            raise ValueError(f"context_prompt_path must be a pathlib.Path")
        
        with self.context_prompt_path.open('r', encoding='utf-8') as f:
            return self.context_prompt_path.read_text(encoding='utf-8')
    
    def _client(self):
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_endpoint
        )
    
    def run(self, base64_image):
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"text",
                            "text": self.context_prompt
                        },
                        {
                            "type":"image_url",
                            "image_url" : {
                                "url":f"data:image/jpg;base64, {base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    


# if __name__ == "__main__":
#     # Instantiate your label helper
#     label = Label()

#     # ⚠️ Printing raw keys is not recommended in production
#     print("[INFO] AOAI_API_KEY:", label.api_key)
#     print("[INFO] AOAI_API_VERSION:", label.api_version)
#     print("[INFO] AOAI_API_ENDPOINT:", label.api_endpoint)
#     print("[INFO] LLM_DEPLOYED_MODEL:", label.llm_model)






    



