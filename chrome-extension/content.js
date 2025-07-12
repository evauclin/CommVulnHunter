// content.js
window.addEventListener('Guardian_Intercepted', (event) => {
    chrome.runtime.sendMessage({
        type: 'ANALYZE_NOTIFICATION',
        data: event.detail
    });
});