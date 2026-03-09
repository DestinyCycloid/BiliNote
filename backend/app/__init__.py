from fastapi import FastAPI

from .routers import note, provider, model, config, redis_monitor



def create_app(lifespan) -> FastAPI:
    app = FastAPI(title="BiliNote",lifespan=lifespan)
    app.include_router(note.router, prefix="/api")
    app.include_router(provider.router, prefix="/api")
    app.include_router(model.router,prefix="/api")
    app.include_router(config.router,  prefix="/api")
    app.include_router(redis_monitor.router, prefix="/api")

    return app
