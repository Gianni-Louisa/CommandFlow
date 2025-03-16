#!/usr/bin/env python3  # Shebang line specifying Python 3 interpreter

"""
# Code Artifact: window_detection.py
# Brief Description: Cross-platform module for detecting open windows and applications to provide context
#                   for speech recognition and command processing systems. ChatGPT assisted in Coding this component.
#
# Programmer: Gianni Louisa
# Date Created: 2/15/2025
# Last Revised: 3/5/2025
#
# Revision History:
#   - 3/15/2025 (Gianni Louisa): Initial creation with Windows support
#
# Preconditions:
#   - Python 3.7+ required
#   - For Windows: pywin32 and psutil packages must be installed
#   - For macOS: pyobjc package must be installed
#   - For Linux: wmctrl and xdotool must be installed on the system
#
# Acceptable Input Values/Types:
#   - For get_context_for_speech_command: Any string containing recognized speech text
#   - For is_application_running: Any string containing an application name
#   - For launch_application: Any string containing an application name
#
# Unacceptable Input Values/Types:
#   - Empty strings or None values may cause unpredictable behavior
#
# Postconditions:
#   - Returns window and application information in structured dictionaries
#   - May launch applications if specifically requested
#
# Return Values:
#   - get_window_snapshot: Dictionary with active window, all windows, and application information
#   - get_context_for_speech_command: Dictionary with context analysis for speech command
#   - is_application_running: Boolean indicating if app is running
#   - launch_application: Boolean indicating success/failure of launch attempt
#
# Error/Exception Conditions:
#   - ImportError: If required OS-specific packages are not installed
#   - NotImplementedError: For unsupported operating systems
#   - Miscellaneous OS-specific errors related to process access or permissions
#
# Side Effects:
#   - Creates log entries through the logging module
#   - May launch applications when using launch_application()
#
# Invariants:
#   - The module will always provide a valid response even if window detection fails
#   - OS detection is guaranteed to work on Windows, macOS, and most Linux desktop environments
#
# Known Faults:
#   - Limited window title detection on macOS requires AppleScript for full functionality
#   - Some Linux window managers may not be fully supported
"""

import platform  # Import platform to identify operating system
import time  # Import time for timing operations
import logging  # Import logging for error and activity tracking
import json  # Import json for structured data handling
import subprocess  # Import subprocess for running external commands
from typing import Dict, List, Optional, Any, Tuple  # Import type hints for better code documentation

# Configure logging system for tracking operations and errors
logging.basicConfig(  # Setup basic configuration for logging
    level=logging.INFO,  # Set log level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Define log message format with timestamp, logger name, level, and message
)
logger = logging.getLogger("WindowDetection")  # Create a logger specific to this module

class WindowDetection:
    """Cross-platform window and application detection system"""  # Class docstring
    
    def __init__(self, polling_interval: float = 0.5):
        """
        Initialize the window detection system
        
        Args:
            polling_interval: How often to check for window changes (in seconds)
        """  # Method docstring with parameter documentation
        self.os_type = platform.system()  # Detect operating system type
        self.polling_interval = polling_interval  # Store polling interval
        self._initialize_os_specific_modules()  # Load OS-specific modules
        self.last_active_window = None  # Initialize storage for previous active window
        self.open_windows = []  # Initialize empty list for storing open windows
        logger.info(f"Window Detection initialized for {self.os_type}")  # Log initialization with OS type
    
    def _initialize_os_specific_modules(self):
        """Load OS-specific libraries for window detection"""  # Method docstring
        if self.os_type == "Windows":  # Check if OS is Windows
            try:
                import win32gui  # Import Windows GUI module
                import win32process  # Import Windows process module
                import psutil  # Import process utilities
                self.win32gui = win32gui  # Store reference to win32gui
                self.win32process = win32process  # Store reference to win32process
                self.psutil = psutil  # Store reference to psutil
                logger.info("Windows modules loaded successfully")  # Log successful module loading
            except ImportError:
                logger.error("Failed to load Windows modules. Please install pywin32 and psutil")  # Log error if import fails
                raise  # Re-raise exception
                
        elif self.os_type == "Darwin":  # Check if OS is macOS
            try:
                # For macOS, we'll use AppKit
                from AppKit import NSWorkspace, NSApplicationActivationPolicyRegular  # Import macOS AppKit modules
                self.NSWorkspace = NSWorkspace  # Store reference to NSWorkspace
                self.NSApplicationActivationPolicyRegular = NSApplicationActivationPolicyRegular  # Store reference to activation policy
                logger.info("macOS modules loaded successfully")  # Log successful module loading
            except ImportError:
                logger.error("Failed to load macOS modules. Please install pyobjc")  # Log error if import fails
                raise  # Re-raise exception
                
        elif self.os_type == "Linux":  # Check if OS is Linux
            try:
                # For Linux, we'll use a combination of tools
                import subprocess  # Import subprocess module
                self.subprocess = subprocess  # Store reference to subprocess
                # Check if wmctrl is installed
                try:
                    subprocess.run(["wmctrl", "-m"], capture_output=True, check=True)  # Try running wmctrl command
                    self.has_wmctrl = True  # Flag that wmctrl is available
                except (subprocess.CalledProcessError, FileNotFoundError):
                    self.has_wmctrl = False  # Flag that wmctrl is not available
                    logger.warning("wmctrl not found, falling back to alternative methods")  # Log warning
                logger.info("Linux modules loaded successfully")  # Log successful module loading
            except ImportError:
                logger.error("Failed to load Linux dependencies")  # Log error if import fails
                raise  # Re-raise exception
        else:
            logger.error(f"Unsupported operating system: {self.os_type}")  # Log error for unsupported OS
            raise NotImplementedError(f"Unsupported operating system: {self.os_type}")  # Raise exception for unsupported OS
    
    def get_active_window(self) -> Dict[str, Any]:
        """
        Get information about the currently active window
        
        Returns:
            Dict containing window information (title, application name, etc.)
        """  # Method docstring with return value documentation
        if self.os_type == "Windows":  # Check if OS is Windows
            return self._get_active_window_windows()  # Return Windows-specific implementation
        elif self.os_type == "Darwin":  # Check if OS is macOS
            return self._get_active_window_macos()  # Return macOS-specific implementation
        elif self.os_type == "Linux":  # Check if OS is Linux
            return self._get_active_window_linux()  # Return Linux-specific implementation
        else:
            return {"error": "Unsupported operating system"}  # Return error for unsupported OS
    
    def _get_active_window_windows(self) -> Dict[str, Any]:
        """Get active window information on Windows"""  # Method docstring
        try:
            hwnd = self.win32gui.GetForegroundWindow()  # Get handle to foreground window
            _, pid = self.win32process.GetWindowThreadProcessId(hwnd)  # Get process ID from window handle
            title = self.win32gui.GetWindowText(hwnd)  # Get window title text
            
            try:
                process = self.psutil.Process(pid)  # Get process object from PID
                app_name = process.name()  # Get application name
                exe_path = process.exe()  # Get executable path
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                app_name = "Unknown"  # Set default name if process can't be accessed
                exe_path = "Unknown"  # Set default path if process can't be accessed
                
            return {
                "title": title,  # Return window title
                "app_name": app_name,  # Return application name
                "pid": pid,  # Return process ID
                "exe_path": exe_path,  # Return executable path
                "hwnd": hwnd,  # Return window handle
                "platform": "windows"  # Return platform identifier
            }
        except Exception as e:
            logger.error(f"Error getting active window on Windows: {e}")  # Log error if exception occurs
            return {"error": str(e), "platform": "windows"}  # Return error information
    
    def _get_active_window_macos(self) -> Dict[str, Any]:
        """Get active window information on macOS"""  # Method docstring
        try:
            workspace = self.NSWorkspace.sharedWorkspace()  # Get shared workspace
            active_app = workspace.frontmostApplication()  # Get frontmost application
            app_name = active_app.localizedName()  # Get localized application name
            pid = active_app.processIdentifier()  # Get process ID
            
            # Get the active window title - note this is more complex on macOS
            # and might require additional tools like AppleScript
            
            return {
                "app_name": app_name,  # Return application name
                "pid": pid,  # Return process ID
                "bundle_id": active_app.bundleIdentifier(),  # Return bundle identifier
                "platform": "macos"  # Return platform identifier
            }
        except Exception as e:
            logger.error(f"Error getting active window on macOS: {e}")  # Log error if exception occurs
            return {"error": str(e), "platform": "macos"}  # Return error information
    
    def _get_active_window_linux(self) -> Dict[str, Any]:
        """Get active window information on Linux"""  # Method docstring
        try:
            if self.has_wmctrl:  # Check if wmctrl is available
                # Use wmctrl to get active window info
                output = self.subprocess.check_output(
                    ["wmctrl", "-l", "-p"],  # Run wmctrl command to list windows with process IDs
                    universal_newlines=True  # Return string output
                )
                
                active_window_output = self.subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowname"],  # Run xdotool to get active window name
                    universal_newlines=True  # Return string output
                ).strip()
                
                # Parse wmctrl output to find matching window
                for line in output.splitlines():  # Iterate through each line of output
                    if active_window_output in line:  # Check if active window name is in the line
                        parts = line.split(None, 4)  # Split line into parts
                        if len(parts) >= 4:  # Ensure we have enough parts
                            window_id, desktop, pid = parts[0], parts[1], parts[2]  # Extract window ID, desktop, and PID
                            title = parts[4] if len(parts) > 4 else ""  # Extract title if available
                            
                            # Get process name
                            try:
                                proc_output = self.subprocess.check_output(
                                    ["ps", "-p", pid, "-o", "comm="],  # Run ps command to get process name
                                    universal_newlines=True  # Return string output
                                ).strip()
                                app_name = proc_output  # Set application name from process name
                            except:
                                app_name = "Unknown"  # Set default name if process can't be accessed
                                
                            return {
                                "title": title,  # Return window title
                                "app_name": app_name,  # Return application name
                                "pid": int(pid),  # Return process ID as integer
                                "window_id": window_id,  # Return window ID
                                "desktop": desktop,  # Return desktop number
                                "platform": "linux"  # Return platform identifier
                            }
            
            # Fallback to using just the active window title
            try:
                title = self.subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowname"],  # Run xdotool to get active window name
                    universal_newlines=True  # Return string output
                ).strip()
                return {"title": title, "platform": "linux"}  # Return title and platform
            except:
                return {"error": "Could not determine active window", "platform": "linux"}  # Return error
                
        except Exception as e:
            logger.error(f"Error getting active window on Linux: {e}")  # Log error if exception occurs
            return {"error": str(e), "platform": "linux"}  # Return error information
    
    def get_all_windows(self) -> List[Dict[str, Any]]:
        """
        Get information about all open windows
        
        Returns:
            List of dictionaries containing window information
        """  # Method docstring with return value documentation
        if self.os_type == "Windows":  # Check if OS is Windows
            return self._get_all_windows_windows()  # Return Windows-specific implementation
        elif self.os_type == "Darwin":  # Check if OS is macOS
            return self._get_all_windows_macos()  # Return macOS-specific implementation
        elif self.os_type == "Linux":  # Check if OS is Linux
            return self._get_all_windows_linux()  # Return Linux-specific implementation
        else:
            return [{"error": "Unsupported operating system"}]  # Return error for unsupported OS
    
    def _get_all_windows_windows(self) -> List[Dict[str, Any]]:
        """Get all open windows on Windows"""  # Method docstring
        windows = []  # Initialize empty list for windows
        
        def enum_windows_callback(hwnd, results):  # Define callback function for EnumWindows
            if self.win32gui.IsWindowVisible(hwnd) and self.win32gui.GetWindowText(hwnd):  # Check if window is visible and has a title
                title = self.win32gui.GetWindowText(hwnd)  # Get window title text
                try:
                    _, pid = self.win32process.GetWindowThreadProcessId(hwnd)  # Get process ID from window handle
                    try:
                        process = self.psutil.Process(pid)  # Get process object from PID
                        app_name = process.name()  # Get application name
                        exe_path = process.exe()  # Get executable path
                    except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                        app_name = "Unknown"  # Set default name if process can't be accessed
                        exe_path = "Unknown"  # Set default path if process can't be accessed
                        
                    results.append({
                        "title": title,  # Add window title
                        "app_name": app_name,  # Add application name
                        "pid": pid,  # Add process ID
                        "exe_path": exe_path,  # Add executable path
                        "hwnd": hwnd,  # Add window handle
                        "platform": "windows"  # Add platform identifier
                    })
                except Exception as e:
                    logger.debug(f"Error processing window {title}: {e}")  # Log error if exception occurs
            return True  # Continue enumeration
        
        try:
            self.win32gui.EnumWindows(enum_windows_callback, windows)  # Enumerate all windows
        except Exception as e:
            logger.error(f"Error enumerating windows on Windows: {e}")  # Log error if exception occurs
            return [{"error": str(e), "platform": "windows"}]  # Return error information
            
        return windows  # Return list of windows
    
    def _get_all_windows_macos(self) -> List[Dict[str, Any]]:
        """Get all open windows on macOS"""  # Method docstring
        try:
            workspace = self.NSWorkspace.sharedWorkspace()  # Get shared workspace
            running_apps = workspace.runningApplications()  # Get all running applications
            
            windows = []  # Initialize empty list for windows
            for app in running_apps:  # Iterate through each running application
                # Only include applications with regular activation policy 
                # (excludes background apps)
                if app.activationPolicy() == self.NSApplicationActivationPolicyRegular:  # Check if app has regular activation policy
                    windows.append({
                        "app_name": app.localizedName(),  # Add application name
                        "pid": app.processIdentifier(),  # Add process ID
                        "bundle_id": app.bundleIdentifier(),  # Add bundle identifier
                        "active": app.isActive(),  # Add active status
                        "platform": "macos"  # Add platform identifier
                    })
            return windows  # Return list of windows
        except Exception as e:
            logger.error(f"Error getting all windows on macOS: {e}")  # Log error if exception occurs
            return [{"error": str(e), "platform": "macos"}]  # Return error information
    
    def _get_all_windows_linux(self) -> List[Dict[str, Any]]:
        """Get all open windows on Linux"""  # Method docstring
        try:
            if self.has_wmctrl:  # Check if wmctrl is available
                windows = []  # Initialize empty list for windows
                output = self.subprocess.check_output(
                    ["wmctrl", "-l", "-p"],  # Run wmctrl command to list windows with process IDs
                    universal_newlines=True  # Return string output
                )
                
                for line in output.splitlines():  # Iterate through each line of output
                    parts = line.split(None, 4)  # Split line into parts
                    if len(parts) >= 4:  # Ensure we have enough parts
                        window_id, desktop, pid = parts[0], parts[1], parts[2]  # Extract window ID, desktop, and PID
                        title = parts[4] if len(parts) > 4 else ""  # Extract title if available
                        
                        # Get process name if possible
                        try:
                            proc_output = self.subprocess.check_output(
                                ["ps", "-p", pid, "-o", "comm="],  # Run ps command to get process name
                                universal_newlines=True  # Return string output
                            ).strip()
                            app_name = proc_output  # Set application name from process name
                        except:
                            app_name = "Unknown"  # Set default name if process can't be accessed
                            
                        windows.append({
                            "title": title,  # Add window title
                            "app_name": app_name,  # Add application name
                            "pid": int(pid) if pid.isdigit() else None,  # Add process ID as integer if it's a number
                            "window_id": window_id,  # Add window ID
                            "desktop": desktop,  # Add desktop number
                            "platform": "linux"  # Add platform identifier
                        })
                return windows  # Return list of windows
            else:
                return [{"error": "wmctrl not available", "platform": "linux"}]  # Return error if wmctrl isn't available
        except Exception as e:
            logger.error(f"Error getting all windows on Linux: {e}")  # Log error if exception occurs
            return [{"error": str(e), "platform": "linux"}]  # Return error information
    
    def start_monitoring(self, callback=None):
        """
        Start monitoring for window changes
        
        Args:
            callback: Function to call when active window changes
                     Function signature: callback(window_info, all_windows)
        """  # Method docstring with parameter documentation
        logger.info("Starting window monitoring")  # Log start of monitoring
        try:
            while True:  # Infinite loop
                current_active = self.get_active_window()  # Get current active window
                all_windows = self.get_all_windows()  # Get all open windows
                
                # Check if active window has changed
                if (self.last_active_window is None or  # Check if this is the first check
                    (current_active.get('title') != self.last_active_window.get('title') or  # Check if title changed
                     current_active.get('app_name') != self.last_active_window.get('app_name'))):  # Check if app name changed
                    
                    self.last_active_window = current_active  # Update last active window
                    self.open_windows = all_windows  # Update open windows list
                    
                    if callback:  # Check if callback is provided
                        callback(current_active, all_windows)  # Call callback with active window and all windows
                
                time.sleep(self.polling_interval)  # Sleep for polling interval
        except KeyboardInterrupt:
            logger.info("Window monitoring stopped")  # Log stop of monitoring on keyboard interrupt
        except Exception as e:
            logger.error(f"Error in window monitoring: {e}")  # Log error if exception occurs
    
    def get_application_context(self) -> Dict[str, Any]:
        """
        Get rich context about current applications for CommandFlow integration
        
        Returns:
            Dictionary with structured information about the computing environment
        """  # Method docstring with return value documentation
        active_window = self.get_active_window()  # Get current active window
        all_windows = self.get_all_windows()  # Get all open windows
        
        # Group windows by application
        apps_dict = {}  # Initialize empty dictionary for applications
        for window in all_windows:  # Iterate through each window
            app_name = window.get('app_name')  # Get application name
            if app_name:  # Check if application name exists
                if app_name not in apps_dict:  # Check if application not already in dictionary
                    apps_dict[app_name] = []  # Initialize empty list for application
                apps_dict[app_name].append(window)  # Add window to application's list
        
        # Count windows by application
        app_counts = {app: len(windows) for app, windows in apps_dict.items()}  # Create dictionary of window counts by application
        
        # Get most used applications (by window count)
        top_apps = sorted(app_counts.items(), key=lambda x: x[1], reverse=True)[:5]  # Sort applications by window count and get top 5
        
        return {
            "active_window": active_window,  # Return active window information
            "window_count": len(all_windows),  # Return total window count
            "application_count": len(apps_dict),  # Return application count
            "top_applications": [{"name": app, "window_count": count} for app, count in top_apps],  # Return top applications
            "all_applications": list(apps_dict.keys()),  # Return all application names
            "platform": self.os_type,  # Return platform
            "timestamp": time.time()  # Return current timestamp
        }


# Standalone function to get a snapshot of window information
def get_window_snapshot() -> Dict[str, Any]:
    """
    Get a snapshot of all current window information
    
    Returns:
        Dictionary with active window, all windows, and application information
    """  # Function docstring with return value documentation
    detector = WindowDetection()  # Create WindowDetection instance
    active_window = detector.get_active_window()  # Get active window
    all_windows = detector.get_all_windows()  # Get all windows
    
    # Extract unique application names
    app_names = set()  # Initialize empty set for application names
    for window in all_windows:  # Iterate through each window
        app_name = window.get('app_name')  # Get application name
        if app_name and app_name != "Unknown":  # Check if application name exists and isn't unknown
            app_names.add(app_name)  # Add application name to set
    
    return {
        "active_window": active_window,  # Return active window information
        "all_windows": all_windows,  # Return all windows
        "window_count": len(all_windows),  # Return window count
        "open_applications": list(app_names),  # Return list of application names
        "application_count": len(app_names),  # Return application count
        "os_type": platform.system(),  # Return operating system type
        "timestamp": time.time()  # Return current timestamp
    }

def is_application_running(app_name: str) -> bool:
    """
    Check if a specific application is running
    
    Args:
        app_name: Name of the application to check (case insensitive)
        
    Returns:
        True if the application is running, False otherwise
    """  # Function docstring with parameter and return value documentation
    snapshot = get_window_snapshot()  # Get window snapshot
    app_name_lower = app_name.lower()  # Convert application name to lowercase
    
    # Check in open applications list
    for app in snapshot["open_applications"]:  # Iterate through open applications
        if app_name_lower in app.lower():  # Check if application name matches (case insensitive)
            return True  # Return True if application is running
    
    # Also check window titles as fallback
    for window in snapshot["all_windows"]:  # Iterate through all windows
        title = window.get("title", "").lower()  # Get window title in lowercase
        window_app_name = window.get("app_name", "").lower()  # Get window application name in lowercase
        
        if app_name_lower in title or app_name_lower in window_app_name:  # Check if application name is in title or application name
            return True  # Return True if application is running
            
    return False  # Return False if application is not running

def launch_application(app_name: str) -> bool:
    """
    Launch an application by name (platform specific)
    
    Args:
        app_name: Name of the application to launch
        
    Returns:
        True if launch was attempted, False otherwise
    """  # Function docstring with parameter and return value documentation
    os_type = platform.system()  # Get operating system type
    
    # Check if app is already running
    if is_application_running(app_name):  # Check if application is already running
        logger.info(f"{app_name} is already running")  # Log that application is already running
        return True  # Return True
        
    try:
        if os_type == "Windows":  # Check if OS is Windows
            # Windows launch using start command
            subprocess.Popen(f"start {app_name}", shell=True)  # Launch application using start command
            return True  # Return True
            
        elif os_type == "Darwin":  # Check if OS is macOS
            # macOS launch using open command
            subprocess.Popen(["open", "-a", app_name])  # Launch application using open command
            return True  # Return True
            
        elif os_type == "Linux":  # Check if OS is Linux
            # Linux launch attempt
            subprocess.Popen([app_name.lower()], shell=True)  # Launch application
            return True  # Return True
            
        else:
            logger.error(f"Unsupported operating system: {os_type}")  # Log error for unsupported OS
            return False  # Return False
            
    except Exception as e:
        logger.error(f"Error launching {app_name}: {e}")  # Log error if exception occurs
        return False  # Return False

def get_context_for_speech_command(recognized_text: str) -> Dict[str, Any]:
    """
    Provide enhanced context for speech commands based on open windows
    
    Args:
        recognized_text: The text recognized from speech
        
    Returns:
        Dictionary with command context and window information
    """  # Function docstring with parameter and return value documentation
    # Get current window information
    snapshot = get_window_snapshot()  # Get window snapshot
    active_window = snapshot["active_window"]  # Get active window
    
    # Clean up recognized text
    text_lower = recognized_text.lower().strip()  # Convert recognized text to lowercase and strip whitespace
    
    # Check if it's likely a false "Thanks for watching" recognition
    if "thanks for watching" in text_lower and not "youtube" in text_lower:  # Check if text contains common phrase but not "youtube"
        # Check if there are any video apps running that might cause this
        video_apps = ["youtube", "vlc", "media player", "netflix", "hulu", "video"]  # List of video application names
        has_video_app = any(app.lower() for app in snapshot["open_applications"]  # Check if any video application is running
                           for video_app in video_apps if video_app in app.lower())
        
        return {
            "recognized_text": recognized_text,  # Return recognized text
            "likely_false_positive": True,  # Flag as likely false positive
            "reason": "Common end-phrase detection without video context",  # Provide reason
            "has_video_app_open": has_video_app,  # Indicate if video application is open
            "active_window": active_window,  # Return active window
            "window_count": snapshot["window_count"],  # Return window count
            "command_confidence": "low"  # Set confidence to low
        }
    
    # Check for app opening commands
    if ("open" in text_lower or "launch" in text_lower or "start" in text_lower) and len(text_lower.split()) >= 2:  # Check if text contains app opening command
        # Extract potential app name (everything after "open/launch/start")
        app_words = text_lower.replace("open ", "").replace("launch ", "").replace("start ", "")  # Extract application name
        
        # Special cases for common apps with different executable names
        app_mapping = {  # Mapping of common application names to executable names
            "vs code": "Visual Studio Code",
            "visual studio code": "Visual Studio Code",
            "visual studio": "Visual Studio",
            "chrome": "Google Chrome",
            "firefox": "Firefox",
            "word": "Microsoft Word",
            "excel": "Microsoft Excel",
            "powerpoint": "Microsoft PowerPoint",
        }
        
        app_name = app_mapping.get(app_words, app_words)  # Get mapped application name or use original
        
        # Check if app is already running
        is_running = is_application_running(app_name)  # Check if application is running
        
        return {
            "recognized_text": recognized_text,  # Return recognized text
            "likely_command": "open_application",  # Set likely command
            "app_name": app_name,  # Return application name
            "app_already_running": is_running,  # Indicate if application is already running
            "active_window": active_window,  # Return active window
            "command_confidence": "high"  # Set confidence to high
        }
    
    # Return general context
    return {
        "recognized_text": recognized_text,  # Return recognized text
        "active_window": active_window,  # Return active window
        "open_apps": snapshot["open_applications"][:5],  # Return first 5 open applications
        "window_count": snapshot["window_count"],  # Return window count
        "command_confidence": "medium"  # Set confidence to medium
    }

# Example usage showing ALL windows and helping with speech commands
if __name__ == "__main__":  # Check if script is being run directly
    # Get all window information
    snapshot = get_window_snapshot()  # Get window snapshot
    
    print(f"Active Window: {snapshot['active_window'].get('title', 'Unknown')} - {snapshot['active_window'].get('app_name', 'Unknown')}")  # Print active window information
    print(f"Open Applications ({snapshot['application_count']}): {', '.join(snapshot['open_applications'])}")  # Print open applications
    print(f"Total Open Windows: {snapshot['window_count']}")  # Print total open windows
    
    # Print ALL open windows
    print("\nALL OPEN WINDOWS:")  # Print header
    for i, window in enumerate(snapshot['all_windows'], 1):  # Iterate through windows with index
        print(f"{i}. {window.get('title', 'Unknown')} - {window.get('app_name', 'Unknown')}")  # Print window information
    
    # Example of how to use context to prevent false "Thanks for watching" detections
    test_command = "Thanks for watching!"  # Test command
    context = get_context_for_speech_command(test_command)  # Get context for test command
    
    if context.get("likely_false_positive"):  # Check if context indicates false positive
        print("\nDetected likely false positive:")  # Print header
        print(f"  • Recognized text: '{test_command}'")  # Print recognized text
        print(f"  • Reason: {context['reason']}")  # Print reason
        print("  • Recommendation: Ignore this command")  # Print recommendation
    
    # Test with a genuine command
    test_command2 = "Open Visual Studio Code"  # Test command
    context2 = get_context_for_speech_command(test_command2)  # Get context for test command
    print("\nGenuine command detected:")  # Print header
    print(f"  • Command: '{test_command2}'")  # Print command
    print(f"  • Interpreted as: {context2['likely_command']} - {context2['app_name']}")  # Print interpretation
    print(f"  • App already running: {context2['app_already_running']}")  # Print if app is already running