# -*- coding: utf-8 -*-
r"""
Студийный микрофон в ТРЕЕ (справа снизу). Обработка работает всегда — не зависит
от браузера. Запуск без чёрного окна: venv\Scripts\pythonw.exe tray.py
Меню: включить/выключить, открыть панель, выход.
"""
import threading, time, webbrowser
from PIL import Image, ImageDraw
import pystray
import app  # наш Flask-модуль (app.py)

PORT = 5002

def _run_flask():
    app.app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)

def _icon_img(active):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = (40, 209, 124, 255) if active else (150, 150, 160, 255)
    d.rounded_rectangle([25, 8, 39, 38], radius=7, fill=c)          # капсула мика
    d.arc([18, 16, 46, 46], start=0, end=180, fill=c, width=4)      # дуга-держатель
    d.line([32, 46, 32, 54], fill=c, width=4)                       # ножка
    d.line([24, 55, 40, 55], fill=c, width=4)                       # основание
    return img

_icon = None
def _refresh():
    if _icon:
        _icon.icon = _icon_img(app._status.get("running"))
        _icon.update_menu()

def _open(icon, item):
    webbrowser.open(f"http://localhost:{PORT}")

def _toggle(icon, item):
    if app._status.get("running"):
        app._stop(); app._stop_meter()
    else:
        app.autostart_from_config(force_start=True)
    app.save_config(); _refresh()

def _preset(name):
    def f(icon, item):
        if name in app.PRESETS:
            app._apply(app.PRESETS[name]); app.save_config()
    return f

def _quit(icon, item):
    try:
        app._stop(); app._stop_meter()
    except Exception:
        pass
    icon.stop()

def _status_text(item):
    return "🟢 В эфире" if app._status.get("running") else "⚪ Выключен"

def _toggle_text(item):
    return "■ Выключить" if app._status.get("running") else "▶ Включить"

def _menu():
    return pystray.Menu(
        pystray.MenuItem(_status_text, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(_toggle_text, _toggle, default=True),
        pystray.MenuItem("🎛 Открыть панель", _open),
        pystray.MenuItem("🎚 Пресеты", pystray.Menu(
            pystray.MenuItem("🎙️ NT1 вещание", _preset("nt1_broadcast")),
            pystray.MenuItem("🔥 Тёплый подкаст", _preset("warm_podcast")),
            pystray.MenuItem("⚪ Нейтрально", _preset("flat")),
        )),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("❌ Выход", _quit),
    )

if __name__ == "__main__":
    threading.Thread(target=_run_flask, daemon=True).start()
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/", timeout=1); break
        except Exception:
            time.sleep(0.5)
    app.autostart_from_config(force_start=True)   # сразу включаем обработку из сохранённых настроек
    _icon = pystray.Icon("MicStudio", _icon_img(app._status.get("running")),
                         "Студийный микрофон", _menu())
    _icon.run()
