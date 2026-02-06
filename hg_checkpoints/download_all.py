"""
CktGen 模型一键下载脚本

使用前配置 TOKEN:
1. 环境变量: export HF_TOKEN=hf_xxx
2. 或命令行: ./download_all.sh --token hf_xxx

用法:
  python download_all.py --run        # 执行下载（增量下载，已存在则跳过）
  python download_all.py --dry_run    # 预览模式
  python download_all.py --output_dir /path/to/dir  # 指定下载目录

特性:
  - 保持 Hugging Face 仓库的原始目录结构
  - 增量下载：已存在的文件自动跳过
  - 支持中断后继续下载（断点续传）
"""

import argparse
import os
import time
from pathlib import Path

# 清除所有代理设置，避免连接问题
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
                  'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY']:
    os.environ.pop(proxy_var, None)

# ========== 超时配置（必须在导入 huggingface_hub 之前设置）==========
TIMEOUT = 600  # 超时时间：600秒 = 10分钟
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(TIMEOUT)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

# 现在才导入 huggingface_hub
from huggingface_hub import snapshot_download, HfApi

# 尝试导入 tqdm 进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ========== 配置 ==========
REPO_ID = "Yuxuan-Hou/CktGen-Test"
SCRIPT_DIR = Path(__file__).parent.absolute()  # 脚本所在目录
DEFAULT_OUTPUT_DIR = SCRIPT_DIR  # 下载到 checkpoints 目录
TOKEN = os.environ.get("HF_TOKEN", "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")

# 重试配置
MAX_RETRIES = 5
RETRY_DELAY = 10  # 重试间隔秒数


def get_local_files(local_dir: Path) -> set:
    """获取本地已存在的文件集合（相对路径）"""
    local_files = set()
    if local_dir.exists():
        for f in local_dir.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(local_dir)
                local_files.add(str(rel_path))
    return local_files


def get_repo_files(api, repo_id: str) -> list:
    """获取仓库所有文件列表"""
    return list(api.list_repo_files(repo_id=repo_id, repo_type="model"))


def download_all():
    """增量下载整个仓库，保持原始目录结构"""
    parser = argparse.ArgumentParser(description="CktGen 模型下载脚本")
    parser.add_argument("--run", action="store_true", help="执行下载")
    parser.add_argument("--dry_run", action="store_true", help="预览模式（只显示要下载的文件）")
    parser.add_argument("--token", type=str, default=TOKEN, help="Hugging Face token")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), 
                        help=f"下载目录 (默认: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    
    # 创建目标目录
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=args.token)

    # 获取仓库文件和本地文件
    print("🔍 获取仓库文件列表...")
    repo_files = get_repo_files(api, REPO_ID)
    local_files = get_local_files(output_dir)

    # 计算需要下载的文件（排除 .gitattributes 等配置文件）
    files_to_download = [f for f in repo_files if f not in local_files and not f.startswith('.')]
    files_to_skip = [f for f in repo_files if f in local_files]

    # 预览模式
    if args.dry_run or not args.run:
        print("=" * 60)
        print("🔍 预览模式 - 下载分析")
        print("=" * 60)
        print(f"仓库: {REPO_ID}")
        print(f"保存目录: {output_dir.absolute()}")
        print(f"来源: Hugging Face 官网")
        print("=" * 60)
        print(f"\n📊 文件统计:")
        print(f"   仓库总文件: {len(repo_files)}")
        print(f"   本地已有: {len(local_files)}")
        print(f"   需下载: {len(files_to_download)}")
        print(f"   已跳过: {len(files_to_skip)}")
        print("=" * 60)

        if files_to_skip:
            print(f"\n⏭️ 跳过的文件 ({len(files_to_skip)} 个，已存在):")
            for f in sorted(files_to_skip)[:10]:
                print(f"   ✓ {f}")
            if len(files_to_skip) > 10:
                print(f"   ... 还有 {len(files_to_skip) - 10} 个文件")

        if files_to_download:
            print(f"\n📦 需下载的文件 ({len(files_to_download)} 个):")
            for f in sorted(files_to_download)[:20]:
                print(f"   • {f}")
            if len(files_to_download) > 20:
                print(f"   ... 还有 {len(files_to_download) - 20} 个文件")
        else:
            print("\n✅ 所有文件已存在，无需下载！")

        if not args.run:
            print("\n💡 使用 --run 执行实际下载")

        return True

    # 执行下载
    print("=" * 60)
    print("📦 CktGen Checkpoints Download (增量模式)")
    print("=" * 60)
    print(f"仓库: {REPO_ID}")
    print(f"保存目录: {output_dir.absolute()}")
    print(f"来源: Hugging Face 官网")
    print(f"超时: {TIMEOUT}秒")
    print(f"状态: {len(files_to_skip)} 个已跳过，{len(files_to_download)} 个待下载")
    print("=" * 60)

    if not files_to_download:
        print("\n✅ 所有文件已存在，无需下载！")
        return True

    # 使用重试机制下载
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n🚀 开始下载 (尝试 {attempt}/{MAX_RETRIES})...")
            start_time = time.time()

            # 使用 snapshot_download 保持原始目录结构
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                token=args.token,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                resume_download=True,  # 断点续传
            )

            duration = time.time() - start_time

            # 重新统计
            local_files = get_local_files(output_dir)
            
            # 统计文件大小
            total_size = 0
            file_count = 0
            for f in output_dir.rglob("*"):
                if f.is_file() and not f.name.startswith('.'):
                    total_size += f.stat().st_size
                    file_count += 1
            total_size_mb = total_size / (1024 * 1024)

            print()
            print("=" * 60)
            print(f"✅ 下载完成!")
            print(f"📁 位置: {output_dir.absolute()}")
            print(f"📄 文件: {file_count} 个")
            print(f"💾 大小: {total_size_mb:.1f} MB")
            print(f"⏱️ 耗时: {duration:.1f} 秒")
            print("=" * 60)

            # 显示目录结构
            print("\n📂 目录结构:")
            for item in sorted(output_dir.iterdir()):
                if item.name.startswith('.'):
                    continue
                if item.is_dir():
                    sub_files = list(item.rglob("*"))
                    sub_file_count = len([f for f in sub_files if f.is_file()])
                    print(f"   📁 {item.name}/ ({sub_file_count} 文件)")
                else:
                    size_mb = item.stat().st_size / 1024 / 1024
                    print(f"   📄 {item.name} ({size_mb:.1f} MB)")

            return True

        except KeyboardInterrupt:
            print("\n\n⚠️ 下载被中断！")
            print("💡 重新运行脚本可以继续下载（已下载的文件会自动跳过）")
            return False
            
        except Exception as e:
            print(f"\n⚠️ 尝试 {attempt} 失败: {e}")
            if attempt < MAX_RETRIES:
                print(f"💤 等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n❌ 下载失败 (已重试 {MAX_RETRIES} 次)")
                print("💡 请检查网络连接后重新运行脚本")
                return False

    return False


if __name__ == "__main__":
    download_all()
