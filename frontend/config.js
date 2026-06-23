// KaggleAgent.AI Configuration
// This file can be used to set environment-specific variables.
// In production, you can replace these values or set them via your hosting provider.

const CONFIG = {
    // API URL for the backend service (Render)
    // Localhost for development, Render URL for production
    API_BASE: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? 'http://localhost:8000'
        : 'https://kaggleagent.onrender.com', // ✅ Render Backend URL
    // Frontend deployment URL
    FRONTEND_URL: 'https://kaggle-agent.vercel.app',
};
