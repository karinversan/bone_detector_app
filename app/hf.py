from __future__ import annotations

from pathlib import Path


def ensure_weights_exist(weights_path: Path, model_label: str) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{model_label} weights not found at: {weights_path}. "
            "Place the weights at the default path in the project."
        )


def maybe_download_weights(
    weights_path: Path, repo_id: str, filename: str, model_label: str
) -> Path:
    if weights_path.exists():
        return weights_path
    if not repo_id:
        raise FileNotFoundError(
            f"{model_label} weights not found at: {weights_path}. "
            "Set HF_REPO_ID env var or place the weights locally."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it to download weights."
        ) from exc

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=weights_path.parent,
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def ensure_file_from_hf(
    local_path: Path,
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
) -> Path:
    if local_path.exists():
        return local_path
    if not repo_id:
        return local_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it to download files."
        ) from exc
    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=local_path.parent,
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def upload_file_to_hf(
    repo_id: str,
    filename: str,
    file_path: Path,
    repo_type: str = "dataset",
    commit_message: str | None = None,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it to upload files."
        ) from exc
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message or f"Update {filename}",
    )
