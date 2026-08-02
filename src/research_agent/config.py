"""Settings loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=str(PROJECT_ROOT / ".env"), extra="ignore")

    groq_api_key: str = ""
    research_agent_model: str = "llama-3.3-70b-versatile"
    research_agent_max_papers: int = 3
    research_agent_critic_threshold: float = 7.0
    research_agent_max_reflection_rounds: int = 2
    research_agent_sandbox_timeout: int = 30
    research_agent_enable_experiments: bool = True
    research_agent_max_autonomous_iterations: int = 2
    research_agent_web_host: str = "127.0.0.1"
    research_agent_web_port: int = 8000

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def papers_dir(self) -> Path:
        path = self.data_dir / "papers"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_dir(self) -> Path:
        path = self.data_dir / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def graph_db_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "graph.db"

    @property
    def outputs_dir(self) -> Path:
        path = PROJECT_ROOT / "outputs" / "summaries"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sessions_dir(self) -> Path:
        path = self.data_dir / "sessions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sandbox_dir(self) -> Path:
        path = self.data_dir / "sandbox"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_db_path(self) -> Path:
        path = self.data_dir / "checkpoints.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
