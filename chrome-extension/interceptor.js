// interceptor.js
(() => {
  'use strict';
  if (window.Notification.isGuardianHijacked) return;
  const OriginalNotification = window.Notification;
  window.Notification = function (title, options) {
    window.dispatchEvent(new CustomEvent('Guardian_NotificationIntercepted', {
      detail: {
        type: 'INTERCEPTED_NOTIFICATION',
        data: { title: title, body: options?.body || '', origin: window.location.origin }
      }
    }));
    return {};
  };
  Object.assign(window.Notification, OriginalNotification);
  window.Notification.isGuardianHijacked = true;
})();