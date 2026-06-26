// KaggleAgent.AI Configuration
// This file can be used to set environment-specific variables.
// In production, you can replace these values or set them via your hosting provider.

const CONFIG = {
    // API URL for the backend service. 
    // Replace the production URL below with your actual deployed backend URL (e.g., Render, Railway)
    API_BASE: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:')
        ? 'http://localhost:8000'
        : 'https://your-production-backend-url.onrender.com', // <-- REPLACE THIS BEFORE DEPLOYING
        
    // Frontend deployment URL
    FRONTEND_URL: window.location.origin,
};
