# ReagentAI Automation Guide

Complete guide to the automated development workflow for ReagentAI in VS Code + GitHub.

## 🚀 Quick Start

### One-Time Setup

1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/TanishikaU-2451/ReagentAI.git
   cd ReagentAI
   chmod +x setup-dev.sh
   ./setup-dev.sh
   ```

2. **Add your API tokens to `.env`:**
   ```bash
   HUGGINGFACE_API_TOKEN=hf_your_token_here
   GITHUB_TOKEN=ghp_your_token_here
   ```

3. **Open in VS Code:**
   ```bash
   code .
   ```

### Daily Development

1. **Start full stack:** `Ctrl+Shift+P` → `Tasks: Run Task` → `Run Full Stack`
2. **Make changes** with automatic formatting and linting
3. **Commit:** `Ctrl+Shift+P` → `Tasks: Run Task` → `Git: Commit & Push`

---

## 🛠 Automation Features

### 1. VS Code Integration

#### Debug Configurations (F5)
- **Backend: FastAPI Debug** - Debug the backend with hot reload
- **Frontend: Next.js Debug** - Debug the frontend with browser auto-open
- **Full Stack** - Debug both simultaneously
- **Test: Backend Pytest** - Run tests with coverage
- **Docker: Full Stack** - Debug containerized version

#### Tasks (Ctrl+Shift+P → Tasks)
- **Run Full Stack** - Start backend + frontend servers
- **Install Dependencies** - Setup Python + Node.js deps
- **Run Tests** - Execute test suites with coverage
- **Lint & Format** - Auto-format all code
- **Git: Commit & Push** - Automated git workflow
- **Docker Compose Up** - Start via Docker

#### Automatic Formatting
- **Python**: Black formatter on save
- **TypeScript/JavaScript**: Prettier formatter on save
- **Import sorting**: Automatic import organization
- **Linting**: Real-time error detection

### 2. GitHub Actions CI/CD

#### On Every Push/PR
```yaml
✅ Backend Testing (pytest, coverage)
✅ Frontend Testing (Jest, type checking)
✅ Code Quality (Black, ESLint, Prettier)
✅ Security Scanning (Bandit, Trivy)
✅ Docker Build & Push
```

#### On Main Branch Push
```yaml
🚀 Deploy Backend (Railway)
🚀 Deploy Frontend (Vercel)
📦 Create GitHub Release
📊 Update Coverage Reports
```

#### Weekly Automated
```yaml
🔄 Dependency Updates (Dependabot-style)
🔒 Security Patch PRs
📈 Performance Reports
```

### 3. Git Workflow Automation

#### Pre-Commit Hooks
```bash
# Runs automatically on 'git commit'
✅ Code formatting (Black, Prettier)
✅ Linting (Flake8, ESLint)
✅ Type checking (MyPy, TypeScript)
✅ Security scanning (Bandit)
✅ Test execution (fast tests only)
✅ Commit message validation
```

#### Conventional Commits
- Interactive commit message wizard
- Automatic changelog generation
- Semantic version bumping
- Release notes generation

---

## 📋 Available Commands

### VS Code Tasks

| Task | Command | Description |
|------|---------|-------------|
| Full Stack | `Ctrl+Shift+P` → `Tasks: Run Task` → `Run Full Stack` | Start backend + frontend |
| Install Deps | `Ctrl+Shift+P` → `Tasks: Run Task` → `Setup Development Environment` | Install all dependencies |
| Run Tests | `Ctrl+Shift+P` → `Tasks: Run Task` → `Run Tests` | Execute test suites |
| Format Code | `Ctrl+Shift+P` → `Tasks: Run Task` → `Lint & Format` | Auto-format all code |
| Git Workflow | `Ctrl+Shift+P` → `Tasks: Run Task` → `Git: Commit & Push` | Automated git workflow |

### Terminal Commands

| Purpose | Command | Description |
|---------|---------|-------------|
| **Development** | `npm run dev:backend` | Start backend with hot reload |
| **Development** | `npm run dev:frontend` | Start frontend with hot reload |
| **Development** | `docker-compose up` | Start via Docker |
| **Testing** | `pytest backend/ -v --cov=backend` | Run backend tests |
| **Testing** | `npm test` | Run frontend tests |
| **Quality** | `black backend/` | Format Python code |
| **Quality** | `npm run lint:fix` | Fix frontend linting |
| **Git** | `cz commit` | Interactive conventional commit |
| **Git** | `cz bump` | Bump version and create release |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `F5` | Start debugging (full stack) |
| `Ctrl+F5` | Start without debugging |
| `Ctrl+Shift+F5` | Restart debugging session |
| `Ctrl+`` ` | Open integrated terminal |
| `Ctrl+Shift+P` | Command palette (tasks) |
| `Ctrl+S` | Save + auto-format |

---

## 🔧 Configuration Files

### Backend Configuration
- **`.pre-commit-config.yaml`** - Git pre-commit hooks
- **`pyproject.toml`** - Python tools configuration
- **`pytest.ini`** - Test configuration (in pyproject.toml)
- **`.env`** - Environment variables
- **`requirements.txt`** - Python dependencies

### Frontend Configuration
- **`.prettierrc`** - Code formatting rules
- **`.eslintrc.json`** - Linting rules
- **`jest.config.js`** - Test configuration
- **`jest.setup.js`** - Test environment setup
- **`package.json`** - Node.js dependencies and scripts
- **`tsconfig.json`** - TypeScript configuration

### VS Code Configuration
- **`.vscode/launch.json`** - Debug configurations
- **`.vscode/tasks.json`** - Task automation
- **`.vscode/settings.json`** - Project settings and extensions

### GitHub Actions
- **`.github/workflows/ci-cd.yml`** - Main CI/CD pipeline
- **`.github/workflows/dependabot.yml`** - Dependency updates

---

## 🎯 Development Workflow

### 1. Feature Development
```bash
# Start development environment
code .                    # Open VS Code
F5                       # Start full stack debugging
# Make your changes with automatic formatting

# Test your changes
Ctrl+Shift+P → "Tasks: Run Tests"

# Commit your changes
Ctrl+Shift+P → "Tasks: Git: Commit & Push"
# Follow conventional commit prompts
```

### 2. Code Quality
- **Automatic**: Formatting on save, import sorting, linting
- **Pre-commit**: Quality checks before commits
- **CI/CD**: Comprehensive checks on push
- **Manual**: `Ctrl+Shift+P` → `Tasks: Lint & Format`

### 3. Testing Strategy
- **Unit Tests**: Fast, isolated component tests
- **Integration Tests**: API endpoint and component integration
- **E2E Tests**: Full pipeline testing
- **Coverage**: Minimum 70% backend, 60% frontend

### 4. Deployment
- **Automatic**: Push to main → Deploy to staging → Deploy to production
- **Manual**: GitHub Actions → Run workflow → Deploy
- **Local**: `docker-compose up` for local testing

---

## 📊 Monitoring & Reports

### GitHub Actions Dashboard
- **Build Status**: Green/red build indicators
- **Test Results**: Detailed test reports and coverage
- **Security Scans**: Vulnerability reports and fixes
- **Performance**: Bundle size, test speed tracking

### Local Development
- **Test Coverage**: `htmlcov/index.html` (auto-generated)
- **Linting Reports**: VS Code problems panel
- **Type Checking**: Real-time TypeScript errors
- **Performance**: Next.js build analyzer

### Production Monitoring
- **Railway**: Backend performance metrics
- **Vercel**: Frontend performance and analytics
- **GitHub**: Dependabot security alerts
- **Codecov**: Coverage trend tracking

---

## 🚨 Troubleshooting

### Common Issues

**🔥 Backend won't start**
```bash
# Check Python environment
source venv/bin/activate
pip install -r requirements.txt

# Check environment variables
cat .env | grep TOKEN
```

**🔥 Frontend won't start**
```bash
cd frontend/nextjs_app
npm install
npm run build  # Check for build errors
```

**🔥 Tests failing**
```bash
# Backend tests
pytest backend/ -v -x  # Stop on first failure

# Frontend tests
npm test -- --verbose
```

**🔥 Pre-commit hooks failing**
```bash
# Run manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

### VS Code Issues
- **Extensions not working**: Install recommended extensions
- **Tasks not visible**: Reload window (`Ctrl+Shift+P` → `Developer: Reload Window`)
- **Debugging not working**: Check `.vscode/launch.json` paths

### GitHub Actions Issues
- **Secrets not configured**: Check [GitHub Secrets Guide](docs/GITHUB_SECRETS.md)
- **Build failing**: Check Actions tab for detailed logs
- **Deployment failing**: Verify deployment service tokens

---

## 🎉 Benefits

### Developer Experience
✅ **Zero-config setup** - Everything works out of the box
✅ **Instant feedback** - Real-time code quality and errors
✅ **One-click debugging** - Full stack debugging with F5
✅ **Automatic formatting** - Never worry about code style
✅ **Smart testing** - Only run affected tests
✅ **Git automation** - Conventional commits and releases

### Code Quality
✅ **100% test coverage tracking** - Never ship untested code
✅ **Security scanning** - Catch vulnerabilities early
✅ **Performance monitoring** - Bundle size and speed tracking
✅ **Dependency management** - Automated updates and security patches
✅ **Type safety** - Full TypeScript and Python type checking

### Team Collaboration
✅ **Consistent environment** - Same setup for all developers
✅ **Automated code review** - Quality checks in PRs
✅ **Clear commit history** - Conventional commits and changelogs
✅ **Deployment automation** - Push to deploy workflow
✅ **Documentation** - Auto-generated docs and guides

---

## 💡 Pro Tips

1. **Use VS Code tasks** instead of memorizing commands
2. **Let pre-commit hooks** catch issues before CI/CD
3. **Write tests first** - they'll guide your implementation
4. **Use conventional commits** - they generate beautiful changelogs
5. **Monitor the Actions tab** - catch issues early
6. **Keep .env updated** - automation depends on proper configuration
7. **Use F5 debugging** - it's faster than print statements
8. **Trust the automation** - it's designed to catch issues you might miss

Happy coding! 🚀