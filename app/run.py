import os

import gradio as gr
import uvicorn
from utils.logger import get_logger

from app.api import create_api_app
from app.ui import build_ui

logger = get_logger(__name__)


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    api_app = create_api_app()
    ui = build_ui(api_base_url=api_base_url)

    # Mount the Gradio app into the same FastAPI server.
    gr.mount_gradio_app(api_app, ui, path="/")

    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(api_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

