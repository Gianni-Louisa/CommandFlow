#!/usr/bin/env python3
"""
WindowDetection - A cross-platform module for detecting open windows and applications
Designed to provide context for speech recognition and command processing
"""

import platform
import time
import logging
import json
import subprocess
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WindowDetection")

class WindowDetection:
    """Cross-platform window and application detection system"""
    
    def __init__(self, polling_interval: float = 0.5):
        """
        Initialize the window detection system
        
        Args:
            polling_interval: How often to check for window changes (in seconds)
        """
        self.os_type = platform.system()
        self.polling_interval = polling_interval
        self._initialize_os_specific_modules()
        self.last_active_window = None
        self.open_windows = []
        logger.info(f"Window Detection initialized for {self.os_type}")
    
    def _initialize_os_specific_modules(self):
        """Load OS-specific libraries for window detection"""
        if self.os_type == "Windows":
            try:
                import win32gui
                import win32process
                import psutil
                self.win32gui = win32gui
                self.win32process = win32process
                self.psutil = psutil
                logger.info("Windows modules loaded successfully")
            except ImportError:
                logger.error("Failed to load Windows modules. Please install pywin32 and psutil")
                raise
                
        elif self.os_type == "Darwin":  # macOS
            try:
                # For macOS, we'll use AppKit
                from AppKit import NSWorkspace, NSApplicationActivationPolicyRegular
                self.NSWorkspace = NSWorkspace
                self.NSApplicationActivationPolicyRegular = NSApplicationActivationPolicyRegular
                logger.info("macOS modules loaded successfully")
            except ImportError:
                logger.error("Failed to load macOS modules. Please install pyobjc")
                raise
                
        elif self.os_type == "Linux":
            try:
                # For Linux, we'll use a combination of tools
                import subprocess
                self.subprocess = subprocess
                # Check if wmctrl is installed
                try:
                    subprocess.run(["wmctrl", "-m"], capture_output=True, check=True)
                    self.has_wmctrl = True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    self.has_wmctrl = False
                    logger.warning("wmctrl not found, falling back to alternative methods")
                logger.info("Linux modules loaded successfully")
            except ImportError:
                logger.error("Failed to load Linux dependencies")
                raise
        else:
            logger.error(f"Unsupported operating system: {self.os_type}")
            raise NotImplementedError(f"Unsupported operating system: {self.os_type}")
    
    def get_active_window(self) -> Dict[str, Any]:
        """
        Get information about the currently active window
        
        Returns:
            Dict containing window information (title, application name, etc.)
        """
        if self.os_type == "Windows":
            return self._get_active_window_windows()
        elif self.os_type == "Darwin":
            return self._get_active_window_macos()
        elif self.os_type == "Linux":
            return self._get_active_window_linux()
        else:
            return {"error": "Unsupported operating system"}
    
    def _get_active_window_windows(self) -> Dict[str, Any]:
        """Get active window information on Windows"""
        try:
            hwnd = self.win32gui.GetForegroundWindow()
            _, pid = self.win32process.GetWindowThreadProcessId(hwnd)
            title = self.win32gui.GetWindowText(hwnd)
            
            try:
                process = self.psutil.Process(pid)
                app_name = process.name()
                exe_path = process.exe()
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                app_name = "Unknown"
                exe_path = "Unknown"
                
            return {
                "title": title,
                "app_name": app_name,
                "pid": pid,
                "exe_path": exe_path,
                "hwnd": hwnd,
                "platform": "windows"
            }
        except Exception as e:
            logger.error(f"Error getting active window on Windows: {e}")
            return {"error": str(e), "platform": "windows"}
    
    def _get_active_window_macos(self) -> Dict[str, Any]:
        """Get active window information on macOS"""
        try:
            workspace = self.NSWorkspace.sharedWorkspace()
            active_app = workspace.frontmostApplication()
            app_name = active_app.localizedName()
            pid = active_app.processIdentifier()
            
            # Get the active window title - note this is more complex on macOS
            # and might require additional tools like AppleScript
            
            return {
                "app_name": app_name,
                "pid": pid,
                "bundle_id": active_app.bundleIdentifier(),
                "platform": "macos"
            }
        except Exception as e:
            logger.error(f"Error getting active window on macOS: {e}")
            return {"error": str(e), "platform": "macos"}
    
    def _get_active_window_linux(self) -> Dict[str, Any]:
        """Get active window information on Linux"""
        try:
            if self.has_wmctrl:
                # Use wmctrl to get active window info
                output = self.subprocess.check_output(
                    ["wmctrl", "-l", "-p"], 
                    universal_newlines=True
                )
                
                active_window_output = self.subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowname"], 
                    universal_newlines=True
                ).strip()
                
                # Parse wmctrl output to find matching window
                for line in output.splitlines():
                    if active_window_output in line:
                        parts = line.split(None, 4)
                        if len(parts) >= 4:
                            window_id, desktop, pid = parts[0], parts[1], parts[2]
                            title = parts[4] if len(parts) > 4 else ""
                            
                            # Get process name
                            try:
                                proc_output = self.subprocess.check_output(
                                    ["ps", "-p", pid, "-o", "comm="],
                                    universal_newlines=True
                                ).strip()
                                app_name = proc_output
                            except:
                                app_name = "Unknown"
                                
                            return {
                                "title": title,
                                "app_name": app_name,
                                "pid": int(pid),
                                "window_id": window_id,
                                "desktop": desktop,
                                "platform": "linux"
                            }
            
            # Fallback to using just the active window title
            try:
                title = self.subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowname"], 
                    universal_newlines=True
                ).strip()
                return {"title": title, "platform": "linux"}
            except:
                return {"error": "Could not determine active window", "platform": "linux"}
                
        except Exception as e:
            logger.error(f"Error getting active window on Linux: {e}")
            return {"error": str(e), "platform": "linux"}
    
    def get_all_windows(self) -> List[Dict[str, Any]]:
        """
        Get information about all open windows
        
        Returns:
            List of dictionaries containing window information
        """
        if self.os_type == "Windows":
            return self._get_all_windows_windows()
        elif self.os_type == "Darwin":
            return self._get_all_windows_macos()
        elif self.os_type == "Linux":
            return self._get_all_windows_linux()
        else:
            return [{"error": "Unsupported operating system"}]
    
    def _get_all_windows_windows(self) -> List[Dict[str, Any]]:
        """Get all open windows on Windows"""
        windows = []
        
        def enum_windows_callback(hwnd, results):
            if self.win32gui.IsWindowVisible(hwnd) and self.win32gui.GetWindowText(hwnd):
                title = self.win32gui.GetWindowText(hwnd)
                try:
                    _, pid = self.win32process.GetWindowThreadProcessId(hwnd)
                    try:
                        process = self.psutil.Process(pid)
                        app_name = process.name()
                        exe_path = process.exe()
                    except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                        app_name = "Unknown"
                        exe_path = "Unknown"
                        
                    results.append({
                        "title": title,
                        "app_name": app_name,
                        "pid": pid,
                        "exe_path": exe_path,
                        "hwnd": hwnd,
                        "platform": "windows"
                    })
                except Exception as e:
                    logger.debug(f"Error processing window {title}: {e}")
            return True
        
        try:
            self.win32gui.EnumWindows(enum_windows_callback, windows)
        except Exception as e:
            logger.error(f"Error enumerating windows on Windows: {e}")
            return [{"error": str(e), "platform": "windows"}]
            
        return windows
    
    def _get_all_windows_macos(self) -> List[Dict[str, Any]]:
        """Get all open windows on macOS"""
        try:
            workspace = self.NSWorkspace.sharedWorkspace()
            running_apps = workspace.runningApplications()
            
            windows = []
            for app in running_apps:
                # Only include applications with regular activation policy 
                # (excludes background apps)
                if app.activationPolicy() == self.NSApplicationActivationPolicyRegular:
                    windows.append({
                        "app_name": app.localizedName(),
                        "pid": app.processIdentifier(),
                        "bundle_id": app.bundleIdentifier(),
                        "active": app.isActive(),
                        "platform": "macos"
                    })
            return windows
        except Exception as e:
            logger.error(f"Error getting all windows on macOS: {e}")
            return [{"error": str(e), "platform": "macos"}]
    
    def _get_all_windows_linux(self) -> List[Dict[str, Any]]:
        """Get all open windows on Linux"""
        try:
            if self.has_wmctrl:
                windows = []
                output = self.subprocess.check_output(
                    ["wmctrl", "-l", "-p"], 
                    universal_newlines=True
                )
                
                for line in output.splitlines():
                    parts = line.split(None, 4)
                    if len(parts) >= 4:
                        window_id, desktop, pid = parts[0], parts[1], parts[2]
                        title = parts[4] if len(parts) > 4 else ""
                        
                        # Get process name if possible
                        try:
                            proc_output = self.subprocess.check_output(
                                ["ps", "-p", pid, "-o", "comm="],
                                universal_newlines=True
                            ).strip()
                            app_name = proc_output
                        except:
                            app_name = "Unknown"
                            
                        windows.append({
                            "title": title,
                            "app_name": app_name,
                            "pid": int(pid) if pid.isdigit() else None,
                            "window_id": window_id,
                            "desktop": desktop,
                            "platform": "linux"
                        })
                return windows
            else:
                return [{"error": "wmctrl not available", "platform": "linux"}]
        except Exception as e:
            logger.error(f"Error getting all windows on Linux: {e}")
            return [{"error": str(e), "platform": "linux"}]
    
    def start_monitoring(self, callback=None):
        """
        Start monitoring for window changes
        
        Args:
            callback: Function to call when active window changes
                     Function signature: callback(window_info, all_windows)
        """
        logger.info("Starting window monitoring")
        try:
            while True:
                current_active = self.get_active_window()
                all_windows = self.get_all_windows()
                
                # Check if active window has changed
                if (self.last_active_window is None or 
                    (current_active.get('title') != self.last_active_window.get('title') or
                     current_active.get('app_name') != self.last_active_window.get('app_name'))):
                    
                    self.last_active_window = current_active
                    self.open_windows = all_windows
                    
                    if callback:
                        callback(current_active, all_windows)
                
                time.sleep(self.polling_interval)
        except KeyboardInterrupt:
            logger.info("Window monitoring stopped")
        except Exception as e:
            logger.error(f"Error in window monitoring: {e}")
    
    def get_application_context(self) -> Dict[str, Any]:
        """
        Get rich context about current applications for CommandFlow integration
        
        Returns:
            Dictionary with structured information about the computing environment
        """
        active_window = self.get_active_window()
        all_windows = self.get_all_windows()
        
        # Group windows by application
        apps_dict = {}
        for window in all_windows:
            app_name = window.get('app_name')
            if app_name:
                if app_name not in apps_dict:
                    apps_dict[app_name] = []
                apps_dict[app_name].append(window)
        
        # Count windows by application
        app_counts = {app: len(windows) for app, windows in apps_dict.items()}
        
        # Get most used applications (by window count)
        top_apps = sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "active_window": active_window,
            "open_window_count": len(all_windows),
            "active_applications": list(apps_dict.keys()),
            "applications_window_count": app_counts,
            "top_applications": top_apps,
            "os_type": self.os_type,
            "timestamp": time.time()
        }

def get_window_snapshot() -> Dict[str, Any]:
    """
    Get a snapshot of ALL currently open windows and applications
    
    Returns:
        Dictionary containing information about active window and all open windows/applications
    """
    detector = WindowDetection()
    active_window = detector.get_active_window()
    all_windows = detector.get_all_windows()
    
    # Group windows by application
    apps_dict = {}
    for window in all_windows:
        app_name = window.get('app_name')
        if app_name:
            if app_name not in apps_dict:
                apps_dict[app_name] = []
            apps_dict[app_name].append(window)
    
    return {
        "active_window": active_window,
        "all_windows": all_windows,  # This contains ALL open windows, not just one
        "open_applications": list(apps_dict.keys()),
        "application_details": apps_dict,
        "window_count": len(all_windows),
        "application_count": len(apps_dict),
        "os_type": platform.system(),
        "timestamp": time.time()
    }

def is_application_running(app_name: str) -> bool:
    """
    Check if a specific application is running
    
    Args:
        app_name: Name of the application to check (case insensitive)
        
    Returns:
        True if the application is running, False otherwise
    """
    snapshot = get_window_snapshot()
    app_name_lower = app_name.lower()
    
    # Check in open applications list
    for app in snapshot["open_applications"]:
        if app_name_lower in app.lower():
            return True
    
    # Also check window titles as fallback
    for window in snapshot["all_windows"]:
        title = window.get("title", "").lower()
        window_app_name = window.get("app_name", "").lower()
        
        if app_name_lower in title or app_name_lower in window_app_name:
            return True
            
    return False

def launch_application(app_name: str) -> bool:
    """
    Launch an application by name (platform specific)
    
    Args:
        app_name: Name of the application to launch
        
    Returns:
        True if launch was attempted, False otherwise
    """
    os_type = platform.system()
    
    # Check if app is already running
    if is_application_running(app_name):
        logger.info(f"{app_name} is already running")
        return True
        
    try:
        if os_type == "Windows":
            # Windows launch using start command
            subprocess.Popen(f"start {app_name}", shell=True)
            return True
            
        elif os_type == "Darwin":  # macOS
            # macOS launch using open command
            subprocess.Popen(["open", "-a", app_name])
            return True
            
        elif os_type == "Linux":
            # Linux launch attempt
            subprocess.Popen([app_name.lower()], shell=True)
            return True
            
        else:
            logger.error(f"Unsupported operating system: {os_type}")
            return False
            
    except Exception as e:
        logger.error(f"Error launching {app_name}: {e}")
        return False

def get_context_for_speech_command(recognized_text: str) -> Dict[str, Any]:
    """
    Provide enhanced context for speech commands based on open windows
    
    Args:
        recognized_text: The text recognized from speech
        
    Returns:
        Dictionary with command context and window information
    """
    # Get current window information
    snapshot = get_window_snapshot()
    active_window = snapshot["active_window"]
    
    # Clean up recognized text
    text_lower = recognized_text.lower().strip()
    
    # Check if it's likely a false "Thanks for watching" recognition
    if "thanks for watching" in text_lower and not "youtube" in text_lower:
        # Check if there are any video apps running that might cause this
        video_apps = ["youtube", "vlc", "media player", "netflix", "hulu", "video"]
        has_video_app = any(app.lower() for app in snapshot["open_applications"] 
                           for video_app in video_apps if video_app in app.lower())
        
        return {
            "recognized_text": recognized_text,
            "likely_false_positive": True,
            "reason": "Common end-phrase detection without video context",
            "has_video_app_open": has_video_app,
            "active_window": active_window,
            "window_count": snapshot["window_count"],
            "command_confidence": "low"
        }
    
    # Check for app opening commands
    if ("open" in text_lower or "launch" in text_lower or "start" in text_lower) and len(text_lower.split()) >= 2:
        # Extract potential app name (everything after "open/launch/start")
        app_words = text_lower.replace("open ", "").replace("launch ", "").replace("start ", "")
        
        # Special cases for common apps with different executable names
        app_mapping = {
            "vs code": "Visual Studio Code",
            "visual studio code": "Visual Studio Code",
            "visual studio": "Visual Studio",
            "chrome": "Google Chrome",
            "firefox": "Firefox",
            "word": "Microsoft Word",
            "excel": "Microsoft Excel",
            "powerpoint": "Microsoft PowerPoint",
        }
        
        app_name = app_mapping.get(app_words, app_words)
        
        # Check if app is already running
        is_running = is_application_running(app_name)
        
        return {
            "recognized_text": recognized_text,
            "likely_command": "open_application",
            "app_name": app_name,
            "app_already_running": is_running,
            "active_window": active_window,
            "command_confidence": "high"
        }
    
    # Return general context
    return {
        "recognized_text": recognized_text,
        "active_window": active_window,
        "open_apps": snapshot["open_applications"][:5],  # First 5 apps for brevity
        "window_count": snapshot["window_count"],
        "command_confidence": "medium"
    }

# Example usage showing ALL windows and helping with speech commands
if __name__ == "__main__":
    # Get all window information
    snapshot = get_window_snapshot()
    
    print(f"Active Window: {snapshot['active_window'].get('title', 'Unknown')} - {snapshot['active_window'].get('app_name', 'Unknown')}")
    print(f"Open Applications ({snapshot['application_count']}): {', '.join(snapshot['open_applications'])}")
    print(f"Total Open Windows: {snapshot['window_count']}")
    
    # Print ALL open windows
    print("\nALL OPEN WINDOWS:")
    for i, window in enumerate(snapshot['all_windows'], 1):
        print(f"{i}. {window.get('title', 'Unknown')} - {window.get('app_name', 'Unknown')}")
    
    # Example of how to use context to prevent false "Thanks for watching" detections
    test_command = "Thanks for watching!"
    context = get_context_for_speech_command(test_command)
    
    if context.get("likely_false_positive"):
        print("\nDetected likely false positive:")
        print(f"  • Recognized text: '{test_command}'")
        print(f"  • Reason: {context['reason']}")
        print("  • Recommendation: Ignore this command")
    
    # Test with a genuine command
    test_command2 = "Open Visual Studio Code"
    context2 = get_context_for_speech_command(test_command2)
    print("\nGenuine command detected:")
    print(f"  • Command: '{test_command2}'")
    print(f"  • Interpreted as: {context2['likely_command']} - {context2['app_name']}")
    print(f"  • App already running: {context2['app_already_running']}")