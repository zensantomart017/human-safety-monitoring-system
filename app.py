import utils.torch_patch  # Workaround for PyTorch 2.6 weights_only=True default
import uvicorn
from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Human Safety Monitoring System API",
    description="API for Person Detection, Tracking, and PPE Detection",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Human Safety Monitoring System API"}

if __name__ == "__main__":
    import yaml
    
    # Load configuration
    try:
        with open("configs/system_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            host = config.get("api", {}).get("host", "0.0.0.0")
            port = config.get("api", {}).get("port", 8000)
    except Exception:
        host = "0.0.0.0"
        port = 8000
        
    uvicorn.run("app:app", host=host, port=port, reload=True)
