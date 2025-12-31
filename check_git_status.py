#!/usr/bin/env python3
"""
Utility script to check what files will be tracked vs ignored by git
"""

import os
import glob
import subprocess
from pathlib import Path


def get_git_status():
    """Get current git status if repo is initialized"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd='.')
        return result.returncode == 0, result.stdout
    except:
        return False, ""


def check_gitignore_patterns():
    """Check what files match .gitignore patterns"""
    
    print("🔍 Git Status Check for DeepSIF_Transformer")
    print("=" * 50)
    
    # Check if git repo exists
    is_git_repo, git_status = get_git_status()
    
    if not is_git_repo:
        print("⚠️  Not a git repository. Run 'git init' to initialize.")
        print("\n📋 Current .gitignore will exclude these files/directories:")
    else:
        print("✅ Git repository detected")
        print(f"\n📋 Git status summary:")
        if git_status.strip():
            print("   Changes detected:")
            for line in git_status.strip().split('\n'):
                print(f"   {line}")
        else:
            print("   Working tree clean")
    
    # Check common directories and files
    print(f"\n📁 Directory Analysis:")
    
    # Large data directories (should be ignored)
    data_dirs = ['labeled_dataset', 'extracted_data', 'source_collections', 
                'model_result', 'real_data', 'source']
    
    print(f"\n🚫 Data Directories (IGNORED):")
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            try:
                size_mb = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk(data_dir)
                             for filename in filenames) / (1024 * 1024)
                file_count = sum(len(filenames) for _, _, filenames in os.walk(data_dir))
                print(f"   📂 {data_dir}/ - {file_count} files, {size_mb:.1f} MB")
            except:
                print(f"   📂 {data_dir}/ - (unable to calculate size)")
        else:
            print(f"   📂 {data_dir}/ - (not found)")
    
    # Model and log files (should be ignored) 
    print(f"\n🚫 Model/Log Files (IGNORED):")
    model_patterns = ['*.pth', '*.pth.tar', '*.log', '*.mat']
    for pattern in model_patterns:
        files = glob.glob(pattern)
        if files:
            total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / (1024 * 1024)
            print(f"   📄 {pattern} - {len(files)} files, {total_size:.1f} MB")
    
    # Python source files (should be tracked)
    print(f"\n✅ Source Code Files (TRACKED):")
    source_patterns = ['*.py', '*.md', '*.txt', '*.yml', '*.yaml', '.gitignore']
    for pattern in source_patterns:
        files = glob.glob(pattern)
        if files:
            total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / 1024  # KB
            print(f"   📄 {pattern} - {len(files)} files, {total_size:.1f} KB")
    
    # Check for requirements.txt and other important files
    print(f"\n📋 Important Project Files:")
    important_files = ['requirements.txt', 'setup.py', 'README.md', 'LICENSE', '.gitignore']
    for file in important_files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"   ✅ {file} - {size_kb:.1f} KB (tracked)")
        else:
            print(f"   ❓ {file} - not found")


def create_requirements_txt():
    """Create requirements.txt if it doesn't exist"""
    
    if os.path.exists('requirements.txt'):
        print("\n✅ requirements.txt already exists")
        return
    
    print("\n📝 Creating requirements.txt...")
    
    # Basic requirements for the project
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "scipy>=1.7.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "scikit-learn>=1.0.0",
        "h5py>=3.1.0",
        "Pillow>=8.3.0"
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write("# DeepSIF Transformer Requirements\n")
        f.write("# PyTorch and related packages\n")
        for req in requirements:
            f.write(f"{req}\n")
    
    print("✅ Created requirements.txt")


def show_git_commands():
    """Show useful git commands for this project"""
    
    print(f"\n🔧 Useful Git Commands:")
    print(f"=" * 30)
    
    commands = [
        ("Initialize repository", "git init"),
        ("Add source files", "git add *.py *.md .gitignore requirements.txt"),
        ("Check status", "git status"),
        ("See what's ignored", "git status --ignored"),
        ("Commit changes", "git commit -m 'Initial commit: DeepSIF Transformer'"),
        ("Add remote origin", "git remote add origin <your-repo-url>"),
        ("Push to remote", "git push -u origin main"),
        ("Check repo size", "git count-objects -vH")
    ]
    
    for description, command in commands:
        print(f"   {description:20}: {command}")


def main():
    check_gitignore_patterns()
    
    if not os.path.exists('requirements.txt'):
        create_requirements_txt()
    
    show_git_commands()
    
    print(f"\n💡 Tips:")
    print(f"   • Keep data files (*.mat) and models (*.pth) OUT of git")
    print(f"   • Only commit source code, docs, and configuration files")
    print(f"   • Use git LFS for large files if absolutely necessary")
    print(f"   • The .gitignore file will handle most exclusions automatically")


if __name__ == "__main__":
    main()
