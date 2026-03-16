# GitHub Repository Secrets Setup

This document explains how to configure the required secrets in your GitHub repository to enable full CI/CD automation.

## Required Secrets

Navigate to your repository → Settings → Secrets and variables → Actions, then add these secrets:

### Core API Access

| Secret Name | Description | Required | Where to Get |
|-------------|-------------|----------|--------------|
| `HUGGINGFACE_API_TOKEN` | HuggingFace API access token | ✅ **Yes** | [HuggingFace Settings](https://huggingface.co/settings/tokens) |
| `GITHUB_TOKEN` | GitHub API access (auto-provided) | ✅ **Yes** | Automatically available |

### Deployment Services (Optional)

| Secret Name | Description | Required | Where to Get |
|-------------|-------------|----------|--------------|
| `RAILWAY_TOKEN` | Railway deployment token | 🔄 Optional | [Railway Dashboard](https://railway.app/dashboard) |
| `VERCEL_TOKEN` | Vercel deployment token | 🔄 Optional | [Vercel Settings](https://vercel.com/account/tokens) |
| `VERCEL_ORG_ID` | Vercel organization ID | 🔄 Optional | Vercel project settings |
| `VERCEL_PROJECT_ID` | Vercel project ID | 🔄 Optional | Vercel project settings |

### Security & Quality (Optional)

| Secret Name | Description | Required | Where to Get |
|-------------|-------------|----------|--------------|
| `CODECOV_TOKEN` | Codecov integration token | 🔄 Optional | [Codecov](https://codecov.io/settings) |

## Setup Instructions

### 1. HuggingFace API Token (Required)

1. Go to [HuggingFace Token Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name: "ReagentAI-CI"
4. Type: **Read**
5. Copy the token (starts with `hf_`)
6. Add to GitHub Secrets as `HUGGINGFACE_API_TOKEN`

**Important**: Accept licenses for gated models:
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [CodeLlama-34b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf)

### 2. Railway Deployment (Optional)

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Connect your GitHub repository
3. Generate a deployment token:
   ```bash
   railway login
   railway environment
   railway variables
   ```
4. Add token as `RAILWAY_TOKEN` secret

### 3. Vercel Deployment (Optional)

1. Install Vercel CLI: `npm i -g vercel`
2. Link your project: `vercel link`
3. Get your tokens:
   ```bash
   vercel env ls
   ```
4. Find IDs in `.vercel/project.json`

### 4. Manual GitHub Secrets Setup

1. Go to your repository on GitHub
2. Click **Settings** tab
3. Navigate to **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add each secret with the exact name from the table above

## Environment Variables in Actions

The CI/CD pipeline uses these secrets as environment variables:

```yaml
env:
  HUGGINGFACE_API_TOKEN: ${{ secrets.HUGGINGFACE_API_TOKEN }}
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
  VERCEL_TOKEN: ${{ secrets.VERCEL_TOKEN }}
```

## Testing the Setup

After adding secrets, push a commit to trigger the CI/CD pipeline:

```bash
git add .
git commit -m "feat: add CI/CD automation"
git push origin main
```

Check the **Actions** tab to see the pipeline run.

## Troubleshooting

### Common Issues

1. **HuggingFace 401 Error**
   - Check token validity: `curl -H "Authorization: Bearer hf_TOKEN" https://huggingface.co/api/whoami`
   - Ensure model licenses are accepted

2. **Railway Deployment Fails**
   - Verify Railway token: `railway whoami`
   - Check service configuration

3. **Vercel Deployment Fails**
   - Verify tokens: `vercel whoami`
   - Check project linking: `vercel link --confirm`

### Debug Commands

```bash
# Test HuggingFace connection
curl -H "Authorization: Bearer $HUGGINGFACE_API_TOKEN" \
  https://huggingface.co/api/models/microsoft/DialoGPT-medium

# Test Railway connection
railway status

# Test Vercel connection
vercel --version
```

## Security Notes

- Never commit secrets to code
- Rotate tokens regularly (every 6 months)
- Use principle of least privilege (read-only when possible)
- Monitor secret usage in GitHub Actions logs
- Consider using GitHub Apps for enhanced security

## Automation Features Enabled

With secrets configured, you get:

✅ **Automated Testing**: Backend + Frontend test suites
✅ **Code Quality**: Linting, formatting, type checking
✅ **Security Scanning**: Dependency and container scanning
✅ **Automated Deployment**: Push-to-deploy workflows
✅ **Dependency Updates**: Automated PR creation
✅ **Release Management**: Semantic versioning

Happy automating! 🚀