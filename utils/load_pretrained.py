#!/usr/bin/env python3
"""
Download CktGen pre-trained models from Hugging Face.

Usage (run from project root):
    # Download all checkpoints
    python utils/load_pretrained.py
    
    # Download specific folder
    python utils/load_pretrained.py --folder cktgen
    python utils/load_pretrained.py --folder evaluator
    python utils/load_pretrained.py --folder baselines
    
    # Download specific baseline model (subfolder)
    python utils/load_pretrained.py --folder baselines/ldt
    python utils/load_pretrained.py --folder baselines/cktgnn
    python utils/load_pretrained.py --folder baselines/pace
    python utils/load_pretrained.py --folder baselines/cvaegan
    
    # List remote files
    python utils/load_pretrained.py --list
    
    # Use mirror (recommended for China)
    HF_ENDPOINT=https://hf-mirror.com python utils/load_pretrained.py
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

# Configuration
REPO_ID = "Yuxuan-Hou/CktGen"
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def get_remote_files(folder=None):
    """Get list of checkpoint files from remote repo."""
    # token=False to disable automatic token reading (repo is public)
    all_files = list_repo_files(REPO_ID, repo_type="model", token=False)
    files = [f for f in all_files if f.endswith(('.pth', '.pkl', '.md'))]
    
    if folder:
        files = [f for f in files if f.startswith(f"{folder}/")]
    
    return files


def download_files(files):
    """Download files to checkpoints directory."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    success, failed = 0, 0
    
    for file_path in files:
        local_path = CHECKPOINTS_DIR / file_path
        
        if local_path.exists():
            print(f"⏭️  {file_path} (exists, skipped)")
            success += 1
            continue
        
        print(f"📥 {file_path}...", end=" ", flush=True)
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type="model",
                local_dir=str(CHECKPOINTS_DIR),
                local_dir_use_symlinks=False,
                token=False,  # Disable auto token (repo is public)
            )
            print("✅")
            success += 1
        except Exception as e:
            print(f"❌ {e}")
            failed += 1
    
    return success, failed


def list_files():
    """List remote files grouped by folder."""
    files = get_remote_files()
    
    folders = {}
    for f in files:
        parts = f.split("/")
        folder = parts[0] if len(parts) > 1 else "root"
        folders.setdefault(folder, []).append(f)
    
    print(f"\n📦 Repository: {REPO_ID}\n")
    for folder, folder_files in sorted(folders.items()):
        print(f"📁 {folder}/")
        for f in folder_files:
            print(f"   └── {f.split('/')[-1]}")
        print()
    
    print(f"Total: {len(files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Download CktGen checkpoints from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    python utils/load_pretrained.py                        # Download all
    python utils/load_pretrained.py --folder cktgen        # Download CktGen models
    python utils/load_pretrained.py --folder evaluator     # Download evaluators
    python utils/load_pretrained.py --folder baselines     # Download all baselines
    python utils/load_pretrained.py --folder baselines/ldt # Download LDT only
    python utils/load_pretrained.py --list                 # List remote files

Files will be downloaded to: {CHECKPOINTS_DIR}
        """
    )
    
    parser.add_argument("--folder", type=str, default=None,
                        help="Download specific folder (e.g., cktgen, baselines, baselines/ldt)")
    parser.add_argument("--list", action="store_true",
                        help="List remote files without downloading")
    
    args = parser.parse_args()
    
    # Check mirror
    mirror = os.environ.get("HF_ENDPOINT")
    if mirror:
        print(f"🪞 Using mirror: {mirror}")
    
    if args.list:
        list_files()
        return
    
    # Get files to download
    files = get_remote_files(args.folder)
    
    if not files:
        print(f"❌ No files found" + (f" in folder '{args.folder}'" if args.folder else ""))
        return
    
    print(f"\n📦 CktGen Checkpoint Downloader")
    print(f"{'='*50}")
    print(f"🔗 Repository: https://huggingface.co/{REPO_ID}")
    print(f"📁 Target: {CHECKPOINTS_DIR}")
    print(f"📥 Files: {len(files)}" + (f" (folder: {args.folder})" if args.folder else " (all)"))
    print(f"{'='*50}\n")
    
    # Download
    success, failed = download_files(files)
    
    # Clean up .cache directory created by huggingface_hub
    cache_dir = CHECKPOINTS_DIR / ".cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    
    print(f"\n{'='*50}")
    print(f"✅ Complete: {success} success, {failed} failed")
    print(f"📁 Location: {CHECKPOINTS_DIR}")


if __name__ == "__main__":
    main()
