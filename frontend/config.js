// KaggleAgent.AI Configuration
// This file can be used to set environment-specific variables.
// In production, you can replace these values or set them via your hosting provider.

const CONFIG = {
    // API URL for the backend service
    // Default to localhost for development
    API_BASE: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? 'http://localhost:8000'
        : 'https://your-appwrite-function-or-render-url.com', // Update this for production
};
