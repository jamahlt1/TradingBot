from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, strategies, accounts, trades, analytics

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(strategies.router)
app.include_router(accounts.router)
app.include_router(trades.router)
app.include_router(analytics.router)

@app.get("/")
def root():
    return {"message": "Backend is running."}