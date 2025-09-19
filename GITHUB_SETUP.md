# 🚀 GitHub Setup Guide

This guide will help you push your Hawkes Process Causality Detection project to GitHub.

## 📋 Prerequisites

1. **GitHub Account**: Create one at [github.com](https://github.com) if you don't have one
2. **Git Installed**: Check with `git --version`
3. **SSH Keys** (recommended) or HTTPS authentication set up

## 🛠️ Step-by-Step Setup

### 1. Initialize Git Repository

```bash
cd /home/hp/F
git init
```

### 2. Configure Git (if not done globally)

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3. Add Files to Git

```bash
# Add all files
git add .

# Or add selectively
git add README.md LICENSE requirements.txt
git add *.py
git add tests/ docs/
```

### 4. Create Initial Commit

```bash
git commit -m "Initial commit: Hawkes Process Causality Detection

- Complete MATLAB to Python conversion
- Branching and thinning simulation methods
- MLE learning with ADMM optimization
- Comprehensive test suite
- Documentation and examples"
```

### 5. Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click "New repository" (green button)
3. Repository name: `hawkes-process-causality`
4. Description: `Python implementation for learning Granger causality in Hawkes processes`
5. Choose **Public** (recommended for open source)
6. **Don't** initialize with README (we already have files)
7. Click "Create repository"

### 6. Add Remote Origin

Replace `yourusername` with your actual GitHub username:

```bash
# SSH (recommended)
git remote add origin git@github.com:yourusername/hawkes-process-causality.git

# OR HTTPS
git remote add origin https://github.com/yourusername/hawkes-process-causality.git
```

### 7. Push to GitHub

```bash
# Push main branch
git branch -M main
git push -u origin main
```

## 📁 Repository Structure

Your GitHub repository will have this structure:

```
hawkes-process-causality/
├── 📄 README.md                    # Project overview & usage
├── 📄 LICENSE                      # MIT license
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 CHANGELOG.md                 # Version history
├── 📄 .gitignore                   # Files to ignore
├── 🐍 __init__.py                  # Package initialization
├── 🐍 test_causality.py           # Main analysis script
├── 🐍 simulation_branch_hp.py     # Simulation methods
├── 🐍 learning_mle_basis.py       # Learning algorithms
├── 🐍 [other Python files]
├── 📁 tests/                       # Test scripts
│   ├── simple_test.py
│   ├── minimal_demo.py
│   ├── comprehensive_test.py
│   └── quick_test.py
└── 📁 docs/                        # Documentation
    └── CONVERSION_SUMMARY.md
```

## 🔧 Additional Setup (Optional)

### Enable GitHub Actions (CI/CD)

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.7, 3.8, 3.9, "3.10", 3.11]

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python tests/simple_test.py
          python tests/comprehensive_test.py
```

### Add Repository Topics

In your GitHub repository:

1. Click ⚙️ "Settings"
2. Scroll to "Topics"
3. Add: `hawkes-process`, `granger-causality`, `point-process`, `machine-learning`, `python`, `time-series`

### Create Release

1. Go to "Releases" tab
2. Click "Create a new release"
3. Tag: `v1.0.0`
4. Title: `Initial Release - v1.0.0`
5. Description: Copy from CHANGELOG.md

## 🎯 Quick Commands Summary

```bash
# Navigate to project
cd /home/hp/F

# Initialize and setup Git
git init
git add .
git commit -m "Initial commit: Hawkes Process Causality Detection"

# Add remote (replace yourusername)
git remote add origin git@github.com:yourusername/hawkes-process-causality.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 🔗 After Pushing

Your repository will be available at:
`https://github.com/yourusername/hawkes-process-causality`

### Share Your Work

1. **Update README**: Replace `yourusername` with your actual username
2. **Add badges**: GitHub will generate URLs for badges
3. **Write examples**: Add more usage examples
4. **Documentation**: Consider adding more detailed docs

## 🎉 Success!

Once pushed, your repository will feature:

- ✅ Complete working Python implementation
- ✅ Comprehensive documentation
- ✅ Test suite with validation
- ✅ Professional project structure
- ✅ Open source license
- ✅ Easy installation via pip

## 🚨 Troubleshooting

### Permission Denied

```bash
# Setup SSH keys
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub
# Copy output and add to GitHub SSH keys
```

### Large Files

If you have large files, use Git LFS:

```bash
git lfs track "*.mat"
git add .gitattributes
```

### Authentication Issues

```bash
# Use personal access token for HTTPS
# GitHub Settings > Developer settings > Personal access tokens
```

Happy coding! 🎉
