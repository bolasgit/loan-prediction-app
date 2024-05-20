from fastapi import FastAPI
from starlette import status
from starlette.responses import RedirectResponse
from routers import predict
from starlette.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/predict", status_code=status.HTTP_302_FOUND)


app.include_router(predict.router)
