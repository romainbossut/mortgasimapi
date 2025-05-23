# Vercel Deployment Guide

This guide will help you deploy your FastAPI Mortgage Simulation API to Vercel.

## Prerequisites

1. [Vercel account](https://vercel.com)
2. [Vercel CLI](https://vercel.com/docs/cli) (optional but recommended)
3. Git repository (GitHub, GitLab, or Bitbucket)

## Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to a Git repository** (if not already done)
   ```bash
   git add .
   git commit -m "Add Vercel configuration"
   git push origin main
   ```

2. **Import your project in Vercel**
   - Go to [vercel.com](https://vercel.com) and sign in
   - Click "Add New..." → "Project"
   - Import your Git repository
   - Vercel will automatically detect the configuration

3. **Deploy**
   - Click "Deploy"
   - Vercel will build and deploy your FastAPI application

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```

## Configuration Files Created

- **`vercel.json`**: Main configuration file that tells Vercel how to build and deploy your FastAPI app
- **`api/index.py`**: Entry point for Vercel's serverless functions
- **`.vercelignore`**: Excludes unnecessary files from deployment

## API Endpoints

Once deployed, your API will be available at `https://your-project.vercel.app` with the following endpoints:

- `GET /` - Health check
- `POST /simulate` - Run mortgage simulation
- `POST /simulate/csv` - Export simulation as CSV
- `GET /simulate/sample` - Get sample request
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## Testing Your Deployment

After deployment, test your API:

1. **Health check**:
   ```bash
   curl https://your-project.vercel.app/
   ```

2. **View API documentation**:
   Visit `https://your-project.vercel.app/docs`

3. **Run a sample simulation**:
   ```bash
   curl -X POST "https://your-project.vercel.app/simulate" \
     -H "Content-Type: application/json" \
     -d @sample_request.json
   ```

## Important Notes

- **Cold starts**: Serverless functions may have a slight delay on first request
- **Timeouts**: Functions have a 30-second maximum execution time
- **File uploads**: Not recommended for large file operations in serverless environment
- **State**: No persistent state between requests

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are in `requirements.txt`
2. **Timeout errors**: Reduce computation complexity or optimize algorithms
3. **Memory issues**: Consider using Vercel Pro for higher memory limits

### Debug Steps

1. Check Vercel function logs in the dashboard
2. Test locally with `vercel dev`
3. Verify all imports work in the serverless environment

## Environment Variables

If you need environment variables, add them in:
- Vercel Dashboard → Project Settings → Environment Variables
- Or use `.env` files (not recommended for production secrets)

## Custom Domain

To use a custom domain:
1. Go to your project in Vercel Dashboard
2. Navigate to Settings → Domains
3. Add your custom domain

---

Your FastAPI Mortgage Simulation API is now ready for deployment to Vercel! 