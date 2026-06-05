import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from model.model import mini_gpt
from model.tokenizer import ByteTokenizer

app = FastAPI(title="Local Decoder API")

class ModelService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "checkpoints", "pretrain.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        cfg = ckpt["model_cfg"]
        self.model = mini_gpt(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            max_seq_len=cfg["max_seq_len"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            num_layers=cfg["num_layers"]
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = ByteTokenizer()
        self.ctx_len = cfg["max_seq_len"]

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        try:
            input_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    input_cond = input_tensor[:, -self.ctx_len:]
                    logits, _ = self.model(input_cond)
                    next_logits = logits[:, -1, :]
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                    input_tensor = torch.cat((input_tensor, next_token), dim=1)
                    if next_token.item() == self.tokenizer.eos_id:
                        break
                        
            generated_ids = input_tensor[0].tolist()
            return self.tokenizer.decode(generated_ids)
        except Exception as e:
            raise RuntimeError(f"error: {str(e)}")

model_service = ModelService()

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

@app.post("/generate")
async def generate_text(request: ChatRequest):
    try:
        response_text = model_service.generate(
            prompt=request.prompt, 
            max_new_tokens=request.max_tokens
        )
        return {"code": 200, "response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)