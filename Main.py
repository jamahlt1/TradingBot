THIS SHOULD BE A LINTER ERRORfrom fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder routers
router_auth = APIRouter(prefix="/auth", tags=["auth"])
router_strategies = APIRouter(prefix="/strategies", tags=["strategies"])
router_accounts = APIRouter(prefix="/accounts", tags=["accounts"])
router_trades = APIRouter(prefix="/trades", tags=["trades"])
router_analytics = APIRouter(prefix="/analytics", tags=["analytics"])

@app.get("/")
def read_root():
    return {"message": "Backend is running."}

# Include routers (to be implemented)
app.include_router(router_auth)
app.include_router(router_strategies)
app.include_router(router_accounts)
app.include_router(router_trades)
app.include_router(router_analytics)
