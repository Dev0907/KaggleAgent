import os
import sys

# Add backend to path so we can import from it
sys.path.append(os.path.join(os.getcwd(), "backend"))

from backend.main import app
import uvicorn

if __name__ == "__main__":
    # Replit uses port 8080 by default for web views sometimes, 
    # but FastAPI standard is 8000. Replit will detect both.
    uvicorn.run(app, host="0.0.0.0", port=8080)
