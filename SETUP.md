# TrueScope ESG Greenwashing Detector - Setup Guide

## Prerequisites

- Node.js 18+ (for both frontend and backend)
- npm or yarn package manager
- OpenAI API key (get one from https://platform.openai.com/api-keys)

## Quick Start

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env  # Or edit .env directly

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here

# Start the backend server
npm run dev
```

The backend will start on `http://localhost:8000` by default.

### 2. Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env  # Or create .env.local

# Edit .env.local and add your configuration
# NEXT_PUBLIC_API_URL=http://localhost:8000
# OPENAI_API_KEY=your_actual_api_key_here (if using server-side API routes)

# Start the frontend development server
npm run dev
```

The frontend will start on `http://localhost:3000` by default.

## Environment Variables

### Backend (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for GPT-4 and embeddings |
| `PORT` | No | Server port (defaults to 8000) |

### Frontend (.env.local)

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | No | Backend API URL (defaults to http://localhost:8000) |
| `OPENAI_API_KEY` | No | Required only if using Next.js API routes instead of standalone backend |

## Architecture

This project has two deployment options:

### Option 1: Standalone Backend + Frontend (Recommended)
- **Backend**: Express.js server running on port 8000
- **Frontend**: Next.js app running on port 3000
- **Communication**: Frontend makes HTTP requests to backend API

### Option 2: Next.js API Routes Only
- **Frontend**: Next.js app with built-in API routes
- **No separate backend needed**
- **Note**: Serverless functions have timeout limits (60s by default)

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found" error**
   - Make sure your `.env` file is in the correct directory
   - Restart the server after adding the API key
   - Check that the key is valid and has the required permissions

2. **"Failed to fetch" or network errors**
   - Ensure the backend is running on port 8000
   - Check that `NEXT_PUBLIC_API_URL` is set correctly in the frontend
   - Verify CORS is enabled (it is by default)

3. **PDF upload fails**
   - Large PDFs (>10MB) may timeout - try compressing first
   - Browser-based extraction is used as fallback for large files
   - Check browser console for detailed error messages

4. **Analysis timeouts**
   - Increase timeout in `next.config.ts` if using Vercel
   - Consider using the standalone backend for large documents
   - Break large reports into smaller chunks

### Development Tips

- Use `npm run dev` for hot-reloading during development
- Check browser console and terminal for error messages
- The in-memory database resets on server restart
- For production, consider using a persistent database like PostgreSQL with pgvector

## API Endpoints

### Backend API (Express)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/health` | Detailed health status |
| POST | `/api/reports` | Upload ESG report |
| GET | `/api/reports` | List all reports |
| POST | `/api/query` | Ask questions about reports |
| POST | `/api/summary` | Generate executive summary |
| POST | `/api/metrics` | Extract ESG metrics |
| POST | `/api/compliance` | Check framework compliance |
| POST | `/api/risk` | Assess greenwashing risk |
| POST | `/api/claims/extract` | Extract claims from report |
| POST | `/api/claims/verify` | Verify claims against evidence |

## Security Notes

- Never commit `.env` files to version control
- Keep your OpenAI API key secure
- Use HTTPS in production
- Implement rate limiting for production deployments
- Add authentication for production use

## Production Deployment

### Backend (Express)

```bash
# Build for production
npm run build  # If using TypeScript

# Start production server
NODE_ENV=production node index.js
```

### Frontend (Next.js)

```bash
# Build for production
npm run build

# Start production server
npm start
```

### Environment-specific Configuration

Set these environment variables in your hosting platform:
- `OPENAI_API_KEY` (required)
- `NODE_ENV=production`
- `NEXT_PUBLIC_API_URL` (frontend only)

## Support

For issues and questions:
1. Check the console logs for error messages
2. Verify environment variables are set correctly
3. Ensure all dependencies are installed
4. Check that ports 3000 and 8000 are available