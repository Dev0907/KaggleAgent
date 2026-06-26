// KaggleAgent.AI Configuration
// This file can be used to set environment-specific variables.
// In production, you can replace these values or set them via your hosting provider.

const CONFIG = {
    // API URL for the backend service (Render)
    // Localhost for development, Render URL for production
    API_BASE: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:')
        ? 'http://localhost:8000'
        : window.location.origin,
    // Frontend deployment URL
    FRONTEND_URL: window.location.origin,
};
