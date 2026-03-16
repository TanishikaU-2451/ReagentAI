#!/bin/bash

# ReagentAI Development Environment Setup Script
# Automates the entire development environment setup

set -e

echo "🚀 Setting up ReagentAI Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking system requirements..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3.11+ is required but not installed."
        exit 1
    fi

    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js 18+ is required but not installed."
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Some features may not work."
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is required but not installed."
        exit 1
    fi

    print_success "System requirements check passed!"
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python environment..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Python virtual environment created!"
    else
        print_warning "Virtual environment already exists."
    fi

    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install Python dependencies
    pip install -r requirements.txt

    # Install development tools
    pip install black flake8 mypy pytest pytest-cov pre-commit

    print_success "Python dependencies installed!"
}

# Setup Node.js environment
setup_nodejs_env() {
    print_status "Setting up Node.js environment..."

    cd frontend/nextjs_app

    # Install dependencies
    npm install

    # Install global tools if needed
    if ! command -v vercel &> /dev/null; then
        npm install -g vercel
    fi

    cd ../..

    print_success "Node.js dependencies installed!"
}

# Setup environment files
setup_env_files() {
    print_status "Setting up environment configuration..."

    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_warning "Created .env file from template. Please add your API tokens:"
        print_warning "  - HUGGINGFACE_API_TOKEN"
        print_warning "  - GITHUB_TOKEN (optional)"
    else
        print_warning ".env file already exists."
    fi

    # Frontend environment
    if [ ! -f "frontend/nextjs_app/.env.local" ]; then
        cp frontend/nextjs_app/.env.local.example frontend/nextjs_app/.env.local
        print_success "Frontend environment file created!"
    fi
}

# Setup pre-commit hooks
setup_git_hooks() {
    print_status "Setting up Git pre-commit hooks..."

    # Source virtual environment
    source venv/bin/activate || source venv/Scripts/activate

    # Install pre-commit
    pre-commit install

    print_success "Pre-commit hooks installed!"
}

# Setup Docker environment
setup_docker() {
    print_status "Setting up Docker environment..."

    if command -v docker &> /dev/null; then
        # Build Docker images
        docker-compose build
        print_success "Docker images built!"
    else
        print_warning "Docker not available. Skipping Docker setup."
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."

    mkdir -p logs
    mkdir -p vector_store
    mkdir -p generated_projects
    mkdir -p docs

    print_success "Project directories created!"
}

# Setup VS Code extensions (if VS Code is available)
setup_vscode() {
    if command -v code &> /dev/null; then
        print_status "Installing recommended VS Code extensions..."

        code --install-extension ms-python.python
        code --install-extension ms-python.black-formatter
        code --install-extension bradlc.vscode-tailwindcss
        code --install-extension esbenp.prettier-vscode
        code --install-extension ms-azuretools.vscode-docker

        print_success "VS Code extensions installed!"
    else
        print_warning "VS Code not found. Skipping extension installation."
    fi
}

# Run tests to verify setup
verify_setup() {
    print_status "Verifying setup..."

    # Source virtual environment
    source venv/bin/activate || source venv/Scripts/activate

    # Test Python imports
    python -c "
import backend.main
import backend.agents
import backend.parser
print('✓ Backend imports work')
"

    # Test frontend build
    cd frontend/nextjs_app
    npm run type-check
    cd ../..

    print_success "Setup verification passed!"
}

# Main setup function
main() {
    echo "🧬 ReagentAI Development Setup"
    echo "============================="

    check_requirements
    create_directories
    setup_env_files
    setup_python_env
    setup_nodejs_env
    setup_git_hooks
    setup_docker
    setup_vscode
    verify_setup

    echo ""
    echo "🎉 Setup Complete!"
    echo ""
    echo "Next steps:"
    echo "1. Add your API tokens to .env file:"
    echo "   - HUGGINGFACE_API_TOKEN=your_token_here"
    echo "   - GITHUB_TOKEN=your_token_here"
    echo ""
    echo "2. Start the development servers:"
    echo "   Backend:  uvicorn backend.main:app --reload"
    echo "   Frontend: cd frontend/nextjs_app && npm run dev"
    echo ""
    echo "3. Or use Docker Compose:"
    echo "   docker-compose up"
    echo ""
    echo "4. Open VS Code to access all automation features:"
    echo "   code ."
    echo ""
    print_success "ReagentAI is ready for development! 🚀"
}

# Run main function
main "$@"