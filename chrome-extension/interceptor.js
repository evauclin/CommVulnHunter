// interceptor.js
(() => {
    if (window.Notification.isHijackedByGuardian) return;

    const OriginalNotification = window.Notification;

    const HijackedNotification = function(title, options) {
        window.dispatchEvent(new CustomEvent('Guardian_Intercepted', {
            detail: {
                title: title,
                body: options?.body || ''
            }
        }));
        return {};
    };

    HijackedNotification.permission = OriginalNotification.permission;
    HijackedNotification.requestPermission = OriginalNotification.requestPermission;
    HijackedNotification.isHijackedByGuardian = true;
    window.Notification = HijackedNotification;
})();