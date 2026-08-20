from fastapi import FastAPI

app = FastAPI(title="Mental Health Chatbot API")

@app.get("/")
def root():
    return {"status": "ok"}
