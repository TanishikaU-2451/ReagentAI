"""ReagentAI Backend Configuration."""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Central configuration loaded from environment variables."""

    # HuggingFace
    huggingface_api_token: str = Field(default="", env="HUGGINGFACE_API_TOKEN")

    # GitHub
    github_token: str = Field(default="", env="GITHUB_TOKEN")

    # Server
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")

    # Docker Sandbox
    sandbox_image: str = Field(default="python:3.11-slim", env="SANDBOX_IMAGE")
    sandbox_timeout: int = Field(default=300, env="SANDBOX_TIMEOUT")
    sandbox_memory_limit: str = Field(default="2g", env="SANDBOX_MEMORY_LIMIT")

    # Paths
    vector_store_path: Path = Field(default=Path("./vector_store"), env="VECTOR_STORE_PATH")
    log_path: Path = Field(default=Path("./logs"), env="LOG_PATH")
    generated_projects_path: Path = Field(
        default=Path("./generated_projects"), env="GENERATED_PROJECTS_PATH"
    )

    # Embedding
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", env="EMBEDDING_MODEL"
    )

    # Agent Models
    research_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="RESEARCH_AGENT_MODEL"
    )
    planning_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="PLANNING_AGENT_MODEL"
    )
    architecture_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="ARCHITECTURE_AGENT_MODEL"
    )
    coding_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="CODING_AGENT_MODEL"
    )
    debug_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="DEBUG_AGENT_MODEL"
    )
    chat_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="CHAT_AGENT_MODEL"
    )
    testing_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="TESTING_AGENT_MODEL"
    )
    validation_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="VALIDATION_AGENT_MODEL"
    )
    diagram_agent_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct", env="DIAGRAM_AGENT_MODEL"
    )

    # Logging
    log_level: str = Field(default="DEBUG", env="LOG_LEVEL")

    # Debug Loop
    max_debug_attempts: int = Field(default=5, env="MAX_DEBUG_ATTEMPTS")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
