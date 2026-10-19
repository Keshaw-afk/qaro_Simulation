"""Take a desktop screenshot via the xdg-desktop-portal (works on GNOME/Wayland once the user has allowed it).
Usage: python3 tools/screenshot.py OUT.png"""
import shutil, sys
from urllib.parse import unquote, urlparse

from gi.repository import Gio, GLib


def screenshot(out, timeout_s=20):
    bus = Gio.bus_get_sync(Gio.BusType.SESSION)
    loop, res = GLib.MainLoop(), {}
    token = "snr%d" % GLib.random_int_range(0, 10 ** 6)
    path = "/org/freedesktop/portal/desktop/request/%s/%s" % (bus.get_unique_name()[1:].replace(".", "_"), token)

    def on_resp(conn, sender, obj, iface, sig, params):
        res["code"], res["data"] = params[0], params[1]
        loop.quit()

    bus.signal_subscribe("org.freedesktop.portal.Desktop", "org.freedesktop.portal.Request", "Response", path, None, Gio.DBusSignalFlags.NONE, on_resp)
    bus.call_sync("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop", "org.freedesktop.portal.Screenshot", "Screenshot",
                  GLib.Variant("(sa{sv})", ("", {"interactive": GLib.Variant("b", False), "handle_token": GLib.Variant("s", token)})), None, Gio.DBusCallFlags.NONE, -1)
    GLib.timeout_add_seconds(timeout_s, loop.quit)
    loop.run()
    if res.get("code") != 0:
        raise RuntimeError(f"portal refused or timed out: {res}")
    shutil.copy(unquote(urlparse(res["data"]["uri"]).path), out)


if __name__ == "__main__":
    screenshot(sys.argv[1])
    print("saved", sys.argv[1])
