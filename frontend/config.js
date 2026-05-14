// KaggleAgent.AI Configuration
// This file can be used to set environment-specific variables.
// In production, you can replace these values or set them via your hosting provider.

const CONFIG = {
    // API URL for the backend service
    // Automatically detects if running on localhost or a production domain like Replit
    API_BASE: (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
        ? 'http://localhost:8000'
        : window.location.origin, // Use the current origin for Replit/Production
};
