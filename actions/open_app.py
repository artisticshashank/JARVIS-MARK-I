import os
import time
import subprocess
import platform
import shutil
from pathlib import Path

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

_SYSTEM = platform.system()

_APP_ALIASES: dict[str, dict[str, str]] = {

    "chrome":             {"Windows": "chrome",                  "Darwin": "Google Chrome",        "Linux": "google-chrome"},
    "google chrome":      {"Windows": "chrome",                  "Darwin": "Google Chrome",        "Linux": "google-chrome"},
    "firefox":            {"Windows": "firefox",                 "Darwin": "Firefox",              "Linux": "firefox"},
    "edge":               {"Windows": "msedge",                  "Darwin": "Microsoft Edge",       "Linux": "microsoft-edge"},
    "brave":              {"Windows": "brave",                   "Darwin": "Brave Browser",        "Linux": "brave-browser"},
    "safari":             {"Windows": "msedge",                  "Darwin": "Safari",               "Linux": "firefox"},
    "opera":              {"Windows": "opera",                   "Darwin": "Opera",                "Linux": "opera"},
    "whatsapp":           {"Windows": "WhatsApp",                "Darwin": "WhatsApp",             "Linux": "whatsapp"},
    "telegram":           {"Windows": "Telegram",                "Darwin": "Telegram",             "Linux": "telegram"},
    "discord":            {"Windows": "Discord",                 "Darwin": "Discord",              "Linux": "discord"},
    "slack":              {"Windows": "Slack",                   "Darwin": "Slack",                "Linux": "slack"},
    "zoom":               {"Windows": "Zoom",                    "Darwin": "zoom.us",              "Linux": "zoom"},
    "teams":              {"Windows": "msteams",                 "Darwin": "Microsoft Teams",      "Linux": "teams"},
    "skype":              {"Windows": "skype",                   "Darwin": "Skype",                "Linux": "skype"},
    "signal":             {"Windows": "signal",                  "Darwin": "Signal",               "Linux": "signal"},
    "spotify":            {"Windows": "Spotify",                 "Darwin": "Spotify",              "Linux": "spotify"},
    "vlc":                {"Windows": "vlc",                     "Darwin": "VLC",                  "Linux": "vlc"},
    "netflix":            {"Windows": "Netflix",                 "Darwin": "Netflix",              "Linux": "firefox"},
    "vscode":             {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "visual studio code": {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "code":               {"Windows": "code",                    "Darwin": "Visual Studio Code",   "Linux": "code"},
    "terminal":           {"Windows": "wt",                      "Darwin": "Terminal",             "Linux": "x-terminal-emulator"},
    "cmd":                {"Windows": "cmd.exe",                 "Darwin": "Terminal",             "Linux": "bash"},
    "powershell":         {"Windows": "powershell.exe",          "Darwin": "Terminal",             "Linux": "bash"},
    "postman":            {"Windows": "Postman",                 "Darwin": "Postman",              "Linux": "postman"},
    "git":                {"Windows": "git-bash",                "Darwin": "Terminal",             "Linux": "bash"},
    "figma":              {"Windows": "Figma",                   "Darwin": "Figma",                "Linux": "figma"},
    "blender":            {"Windows": "blender",                 "Darwin": "Blender",              "Linux": "blender"},
    "word":               {"Windows": "winword",                 "Darwin": "Microsoft Word",       "Linux": "libreoffice --writer"},
    "excel":              {"Windows": "excel",                   "Darwin": "Microsoft Excel",      "Linux": "libreoffice --calc"},
    "powerpoint":         {"Windows": "powerpnt",                "Darwin": "Microsoft PowerPoint", "Linux": "libreoffice --impress"},
    "libreoffice":        {"Windows": "soffice",                 "Darwin": "LibreOffice",          "Linux": "libreoffice"},
    "notepad":            {"Windows": "notepad.exe",             "Darwin": "TextEdit",             "Linux": "gedit"},
    "textedit":           {"Windows": "notepad.exe",             "Darwin": "TextEdit",             "Linux": "gedit"},
    "explorer":           {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "file explorer":      {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "finder":             {"Windows": "explorer.exe",            "Darwin": "Finder",               "Linux": "nautilus"},
    "task manager":       {"Windows": "taskmgr.exe",             "Darwin": "Activity Monitor",     "Linux": "gnome-system-monitor"},
    "settings":           {"Windows": "ms-settings:",            "Darwin": "System Preferences",   "Linux": "gnome-control-center"},
    "calculator":         {"Windows": "calc.exe",                "Darwin": "Calculator",           "Linux": "gnome-calculator"},
    "paint":              {"Windows": "mspaint.exe",             "Darwin": "Preview",              "Linux": "gimp"},
    "instagram":          {"Windows": "Instagram",               "Darwin": "Instagram",            "Linux": "firefox"},
    "tiktok":             {"Windows": "TikTok",                  "Darwin": "TikTok",               "Linux": "firefox"},
    "notion":             {"Windows": "Notion",                  "Darwin": "Notion",               "Linux": "notion"},
    "obsidian":           {"Windows": "Obsidian",                "Darwin": "Obsidian",             "Linux": "obsidian"},
    "capcut":             {"Windows": "CapCut",                  "Darwin": "CapCut",               "Linux": "capcut"},
    "steam":              {"Windows": "steam",                   "Darwin": "Steam",                "Linux": "steam"},
    "epic":               {"Windows": "EpicGamesLauncher",       "Darwin": "Epic Games Launcher",  "Linux": "legendary"},
    "epic games":         {"Windows": "EpicGamesLauncher",       "Darwin": "Epic Games Launcher",  "Linux": "legendary"},
}


def _normalize(raw: str) -> str:
    key = raw.lower().strip()

    if key in _APP_ALIASES:
        return _APP_ALIASES[key].get(_SYSTEM, raw)

    for alias_key, os_map in _APP_ALIASES.items():
        if alias_key in key or key in alias_key:
            return os_map.get(_SYSTEM, raw)

    return raw  

# ── Windows: find it, launch it, then *look* ────────────────────────────────
#
# The old path pressed the Windows key, typed the app's name and returned True.
# Three things were wrong with that, and only the third is obvious:
#
#   * it typed into whatever window had focus once the Start Menu closed, so a
#     slow Start Menu sent the name somewhere else entirely;
#   * it never checked, so "Spotify is open" was reported for an app that had
#     never appeared — the user was told a thing happened and it had not;
#   * an app that is *already running* (Spotify sits in the tray, as most music
#     players do) is not a thing you launch. It is a thing you bring to the
#     front, and pressing Win and typing its name is not how you do that.
#
# So: focus what is running, resolve a real executable for what is not, launch
# it the way Windows itself would, and wait until a process actually exists
# before answering. The keystroke route survives as a last resort only.

_DETACHED = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}


def _win_process_for(app_name: str):
    """The running process for this app, or None.

    Matches the executable's stem as well as its path, because an app called
    "Visual Studio Code" runs as `Code.exe` — the display name and the process
    name are frequently not the same string.
    """
    if not _PSUTIL:
        return None
    want = app_name.lower().replace(" ", "")
    if not want:
        return None
    if want.endswith(".exe"):
        want = want[:-4]

    for proc in psutil.process_iter(["name", "exe"]):
        try:
            name = (proc.info.get("name") or "").lower().replace(" ", "")
            exe = (proc.info.get("exe") or "").lower().replace(" ", "")
        except Exception:
            continue
        if not name:
            continue
        stem = name[:-4] if name.endswith(".exe") else name
        if stem in ("python", "pythonw", "jarvis"):    # never find ourselves
            continue
        if stem == want or want in stem:
            return proc
        if exe and want in exe.rsplit("\\", 1)[-1]:
            return proc
    return None


def _win_focus(app_name: str) -> bool:
    """Raise an already-running app's window. False if it has none to raise."""
    try:
        import pygetwindow as gw
    except Exception:
        return False

    want = app_name.lower().replace(".exe", "")
    try:
        windows = [w for w in gw.getAllWindows()
                   if getattr(w, "title", "") and want in w.title.lower()]
    except Exception:
        return False

    for win in windows:
        try:
            if getattr(win, "isMinimized", False):
                win.restore()
            win.activate()
            time.sleep(0.6)
            return True
        except Exception:
            try:
                # activate() is refused for some windows; the nudge is not.
                win.minimize()
                win.restore()
                time.sleep(0.4)
                return True
            except Exception:
                continue
    return False


def _win_app_paths(app_name: str):
    """The executable registered under App Paths — where an installer declares
    an app so that typing its name into Run works. Spotify is *not* registered
    there, but plenty of things are, and it costs one registry read to check."""
    try:
        import winreg
    except Exception:
        return None

    stem = app_name[:-4] if app_name.lower().endswith(".exe") else app_name
    sub = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for exe in (stem + ".exe", stem):
            try:
                with winreg.OpenKey(root, sub + "\\" + exe) as key:
                    path = winreg.QueryValueEx(key, None)[0]
                if path and Path(path).exists():
                    return str(path)
            except Exception:
                continue
    return None


def _win_common_install(app_name: str):
    """Per-user and per-machine install folders. Most packaged apps land in one
    of these and never touch PATH — which is exactly why `shutil.which` found
    nothing for an app that was installed and running."""
    stem = app_name[:-4] if app_name.lower().endswith(".exe") else app_name
    for root in (os.environ.get("LOCALAPPDATA"), os.environ.get("APPDATA"),
                 os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if not root:
            continue
        for folder in (Path(root) / stem,
                       Path(root) / "Programs" / stem,
                       Path(root) / stem / "Application"):
            for exe in (folder / f"{stem}.exe", folder / f"{stem}Launcher.exe"):
                if exe.exists():
                    return str(exe)
    return None


def _win_start_menu(app_name: str):
    """This user's (or all users') Start Menu shortcut for the app.

    Launching the .lnk with `os.startfile` is what clicking it does — minus the
    window that has to be typed into, which is the whole problem.
    """
    stem = app_name[:-4] if app_name.lower().endswith(".exe") else app_name
    want = stem.lower().replace(" ", "")
    best = None
    for root in (os.environ.get("APPDATA"), os.environ.get("ProgramData")):
        if not root:
            continue
        folder = Path(root) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        if not folder.exists():
            continue
        try:
            for lnk in folder.rglob("*.lnk"):
                name = lnk.stem.lower().replace(" ", "")
                if name == want:
                    return str(lnk)              # an exact name wins outright
                if want in name and best is None:
                    best = str(lnk)
        except Exception:
            continue
    return best


def _win_wait(app_name: str, timeout: float) -> bool:
    """Wait for the process to actually exist.

    This is the fix in one function: "did it open?" is a fact about the machine,
    not an assumption made two seconds after a keystroke.
    """
    end = time.time() + timeout
    while time.time() < end:
        if _win_process_for(app_name) is not None:
            return True
        time.sleep(0.25)
    return False


def _launch_windows(app_name: str) -> bool:

    # 1. Already running — bring it to the front. The commonest case, and the
    #    one that failed hardest: a music player living in the notification area.
    if _win_process_for(app_name) is not None and _win_focus(app_name):
        return True

    # 2. A real executable: PATH, then App Paths, then the usual install dirs.
    target = (shutil.which(app_name)
              or shutil.which(app_name.split(".")[0])
              or _win_app_paths(app_name)
              or _win_common_install(app_name))
    if target:
        try:
            subprocess.Popen([target], **_DETACHED)
            if _win_wait(app_name, 12.0):
                return True
            print(f"[open_app] '{target}' started but no process appeared")
        except Exception as e:
            print(f"[open_app] launch failed for '{target}': {e}")

    # 3. The Start Menu shortcut, launched rather than mimed.
    link = _win_start_menu(app_name)
    if link:
        try:
            os.startfile(link)
            if _win_wait(app_name, 12.0):
                return True
            print(f"[open_app] shortcut did not start a process: {link}")
        except Exception as e:
            print(f"[open_app] could not start '{link}': {e}")

    # 4. URL handlers and ms-settings:-style protocols.
    if ":" in app_name:
        try:
            subprocess.Popen(f"start {app_name}", shell=True, **_DETACHED)
            time.sleep(1.0)
            return True
        except Exception:
            pass

    # 5. Last resort: the Start Menu, typed into. Kept because it is the only
    #    thing left that can reach a Store app with no path and no shortcut we
    #    can see — but it is last now, and it reports what actually happened
    #    instead of quietly returning True.
    try:
        import pyautogui
        pyautogui.PAUSE = 0.1
        pyautogui.press("win")
        time.sleep(0.7)
        pyautogui.write(app_name, interval=0.05)
        time.sleep(0.9)
        pyautogui.press("enter")
        return _win_wait(app_name, 8.0)
    except Exception as e:
        print(f"[open_app] Start Menu search failed: {e}")

    return False


def _launch_macos(app_name: str) -> bool:

    try:
        result = subprocess.run(
            ["open", "-a", app_name],
            capture_output=True, timeout=8
        )
        if result.returncode == 0:
            time.sleep(1.0)
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["open", "-a", f"{app_name}.app"],
            capture_output=True, timeout=8
        )
        if result.returncode == 0:
            time.sleep(1.0)
            return True
    except Exception:
        pass

    binary = shutil.which(app_name) or shutil.which(app_name.lower())
    if binary:
        try:
            subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.0)
            return True
        except Exception:
            pass

    try:
        import pyautogui
        pyautogui.hotkey("command", "space")
        time.sleep(0.6)
        pyautogui.write(app_name, interval=0.05)
        time.sleep(0.8)
        pyautogui.press("enter")
        time.sleep(1.5)
        return True
    except Exception as e:
        print(f"[open_app] Spotlight failed: {e}")

    return False


_LINUX_TERMINAL_FALLBACKS = [
    "x-terminal-emulator", "gnome-terminal", "konsole", "xfce4-terminal",
    "xterm", "lxterminal", "mate-terminal", "tilix", "alacritty", "kitty",
]

def _launch_linux(app_name: str) -> bool:

    # terminal emulators: try common ones in order
    if app_name in ("x-terminal-emulator", "gnome-terminal", "terminal"):
        for term in _LINUX_TERMINAL_FALLBACKS:
            if shutil.which(term):
                try:
                    subprocess.Popen([term], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(1.0)
                    return True
                except Exception:
                    continue

    binary = (
        shutil.which(app_name) or
        shutil.which(app_name.lower()) or
        shutil.which(app_name.lower().replace(" ", "-")) or
        shutil.which(app_name.lower().replace(" ", "_"))
    )
    if binary:
        try:
            subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1.0)
            return True
        except Exception:
            pass

    try:
        subprocess.run(
            ["xdg-open", app_name],
            capture_output=True, timeout=5
        )
        return True
    except Exception:
        pass

    for desktop_name in [
        app_name.lower(),
        app_name.lower().replace(" ", "-"),
        app_name.lower().replace(" ", ""),
    ]:
        try:
            result = subprocess.run(
                ["gtk-launch", desktop_name],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    return False


_OS_LAUNCHERS = {
    "Windows": _launch_windows,
    "Darwin":  _launch_macos,
    "Linux":   _launch_linux,
}

def open_app(
    parameters=None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    app_name = (parameters or {}).get("app_name", "").strip()

    if not app_name:
        return "No application name provided."

    launcher = _OS_LAUNCHERS.get(_SYSTEM)
    if launcher is None:
        return f"Unsupported operating system: {_SYSTEM}"

    normalized = _normalize(app_name)
    print(f"[open_app] Launching: '{app_name}' → '{normalized}' ({_SYSTEM})")

    if player:
        player.write_log(f"[open_app] {app_name}")

    try:
        if launcher(normalized):
            return f"Opened {app_name}."
        if normalized.lower() != app_name.lower():
            if launcher(app_name):
                return f"Opened {app_name}."
        return (
            f"Could not confirm that {app_name} launched. "
            f"It may still be loading, or it might not be installed."
        )
    except Exception as e:
        print(f"[open_app] Error: {e}")
        return f"Failed to open {app_name}: {e}"


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "open_app",
    "description": "Opens any application on the computer. Use this whenever the user asks to open, launch, or start any app, website, or program. Always call this tool — never just say you opened it.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "app_name": {
                "type": "STRING",
                "description": "Exact name of the application (e.g. 'WhatsApp', 'Chrome', 'Spotify')"
            }
        },
        "required": [
            "app_name"
        ]
    },
    "handler": open_app,
}
