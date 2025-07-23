import logging
import os
import json
import time
import uuid
import socket
import platform
import traceback
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import pandas as pd
from collections import defaultdict, deque


class LogLevel(Enum):
    """Log level enumeration to categorize log entries."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"
    METRIC = "METRIC"
    USER = "USER"
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"
    COMPLIANCE = "COMPLIANCE"


class LogCategory(Enum):
    """Category enumeration to organize log entries."""
    ANALYSIS = "ANALYSIS"
    MODEL = "MODEL"
    USER_ACTION = "USER_ACTION"
    SYSTEM = "SYSTEM"
    DATA_PROCESSING = "DATA_PROCESSING"
    FILE_OPERATION = "FILE_OPERATION"
    API = "API"
    DATABASE = "DATABASE"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    COMPLIANCE = "COMPLIANCE"
    EXPORT = "EXPORT"
    IMPORT = "IMPORT"
    ANOMALY = "ANOMALY"
    PREDICTION = "PREDICTION"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    DASHBOARD = "DASHBOARD"
    AUTHENTICATION = "AUTHENTICATION"
    OTHER = "OTHER"


class LoggingSystem:
    """
    Comprehensive logging and audit trail system for tracking 
    user actions, model performance, and system events.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one logging instance exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LoggingSystem, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, log_dir: str = "logs", 
                 retention_days: int = 30, 
                 log_level: LogLevel = LogLevel.INFO,
                 enable_file: bool = True,
                 enable_console: bool = True,
                 enable_audit: bool = True,
                 enable_metrics: bool = True):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory for storing logs
            retention_days: How many days to keep logs
            log_level: Minimum log level to record
            enable_file: Whether to write logs to file
            enable_console: Whether to write logs to console
            enable_audit: Whether to record detailed audit logs
            enable_metrics: Whether to collect performance metrics
        """
        # Check if already initialized
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.log_dir = log_dir
        self.retention_days = retention_days
        self.log_level = log_level
        self.enable_file = enable_file
        self.enable_console = enable_console
        self.enable_audit = enable_audit
        self.enable_metrics = enable_metrics
        
        # Create log directories
        self.general_log_dir = os.path.join(log_dir, "general")
        self.audit_log_dir = os.path.join(log_dir, "audit")
        self.metrics_log_dir = os.path.join(log_dir, "metrics")
        self.error_log_dir = os.path.join(log_dir, "error")
        
        os.makedirs(self.general_log_dir, exist_ok=True)
        os.makedirs(self.audit_log_dir, exist_ok=True)
        os.makedirs(self.metrics_log_dir, exist_ok=True)
        os.makedirs(self.error_log_dir, exist_ok=True)
        
        # Set up Python logging
        self.logger = logging.getLogger("sentiment_analysis_system")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Remove any existing handlers
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler for general logs
        if self.enable_file:
            general_log_file = os.path.join(
                self.general_log_dir,
                f"log_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler = logging.FileHandler(general_log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Error log file handler
            error_log_file = os.path.join(
                self.error_log_dir,
                f"error_{datetime.now().strftime('%Y%m%d')}.log"
            )
            error_handler = logging.FileHandler(error_log_file)
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # In-memory recent logs (for UI display)
        self.recent_logs = deque(maxlen=1000)
        
        # Performance metrics tracking
        self.metrics = defaultdict(list)
        self.metrics_lock = threading.Lock()
        
        # System information
        self.system_info = self._collect_system_info()
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # User information (to be set later)
        self.user_id = "anonymous"
        
        # Start logging system
        self._start_logging_system()
        
        # Clean up old logs
        self._clean_up_old_logs()
        
        self.initialized = True
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collects information about the system.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system()
        }
        
        # Add more details if available
        try:
            import psutil
            info["memory_total"] = psutil.virtual_memory().total
            info["cpu_count"] = psutil.cpu_count()
        except ImportError:
            pass
            
        return info
    
    def _start_logging_system(self):
        """Initializes the logging system and logs startup information."""
        self.log_system_event(
            "Logging system initialized", 
            LogLevel.INFO, 
            LogCategory.SYSTEM, 
            details={
                "log_level": self.log_level.value,
                "log_dir": self.log_dir,
                "retention_days": self.retention_days,
                "session_id": self.session_id
            }
        )
        
        # Log system information
        self.log_system_event(
            "System information", 
            LogLevel.INFO, 
            LogCategory.SYSTEM,
            details=self.system_info
        )
    
    def _clean_up_old_logs(self):
        """Removes log files older than retention_days."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Clean up each log directory
        for log_dir in [self.general_log_dir, self.audit_log_dir, 
                        self.metrics_log_dir, self.error_log_dir]:
            if not os.path.exists(log_dir):
                continue
                
            for filename in os.listdir(log_dir):
                file_path = os.path.join(log_dir, filename)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Check file modification time
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_date:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        # Just log the error but don't raise it
                        self.logger.warning(f"Could not delete old log file {file_path}: {e}")
    
    def log_user_action(self, 
                        action: str, 
                        details: Optional[Dict[str, Any]] = None, 
                        user_id: Optional[str] = None):
        """
        Logs a user action for audit purposes.
        
        Args:
            action: Description of the action
            details: Optional details about the action
            user_id: Optional user ID (defaults to current user)
        """
        if not self.enable_audit:
            return
            
        if user_id:
            self.user_id = user_id
            
        timestamp = datetime.now()
        
        audit_entry = {
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "action": action,
            "details": details or {},
            "level": LogLevel.AUDIT.value,
            "category": LogCategory.USER_ACTION.value
        }
        
        # Add to recent logs
        self.recent_logs.append(audit_entry)
        
        # Log to file
        self._write_log_to_file(audit_entry, "audit")
            
        # Also log to general Python logger
        self.logger.info(f"User action: {action}")
    
    def log_model_performance(self, 
                              model_name: str, 
                              metrics: Dict[str, Any],
                              prediction_details: Optional[Dict[str, Any]] = None):
        """
        Logs model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics
            prediction_details: Optional details about the prediction
        """
        if not self.enable_metrics:
            return
            
        timestamp = datetime.now()
        
        performance_entry = {
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "model_name": model_name,
            "metrics": metrics,
            "prediction_details": prediction_details or {},
            "level": LogLevel.METRIC.value,
            "category": LogCategory.MODEL.value
        }
        
        # Add to recent logs
        self.recent_logs.append(performance_entry)
        
        # Store metrics for analysis
        with self.metrics_lock:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics[f"{model_name}_{key}"].append((timestamp, value))
        
        # Log to file
        self._write_log_to_file(performance_entry, "metrics")
            
        # Log summary to general logger
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(f"Model performance - {model_name}: {metrics_str}")
    
    def log_system_event(self, 
                         event: str, 
                         level: LogLevel, 
                         category: LogCategory,
                         details: Optional[Dict[str, Any]] = None,
                         exception: Optional[Exception] = None):
        """
        Logs a system event.
        
        Args:
            event: Description of the event
            level: Log level
            category: Event category
            details: Optional details about the event
            exception: Optional exception that triggered the event
        """
        timestamp = datetime.now()
        
        # Add exception details if provided
        event_details = details or {}
        if exception:
            event_details["exception_type"] = type(exception).__name__
            event_details["exception_message"] = str(exception)
            event_details["traceback"] = traceback.format_exc()
        
        system_entry = {
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "event": event,
            "level": level.value,
            "category": category.value,
            "details": event_details
        }
        
        # Add to recent logs
        self.recent_logs.append(system_entry)
        
        # Log to file based on level
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._write_log_to_file(system_entry, "error")
        else:
            self._write_log_to_file(system_entry, "general")
            
        # Log to Python logger
        log_method = getattr(self.logger, level.value.lower(), self.logger.info)
        log_method(f"{category.value}: {event}")
    
    def log_data_processing(self, 
                           operation: str, 
                           input_details: Dict[str, Any],
                           output_summary: Dict[str, Any],
                           duration_ms: float):
        """
        Logs data processing operations.
        
        Args:
            operation: Type of operation performed
            input_details: Information about the input data
            output_summary: Summary of the processed output
            duration_ms: Processing duration in milliseconds
        """
        timestamp = datetime.now()
        
        processing_entry = {
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "operation": operation,
            "input_details": input_details,
            "output_summary": output_summary,
            "duration_ms": duration_ms,
            "level": LogLevel.INFO.value,
            "category": LogCategory.DATA_PROCESSING.value
        }
        
        # Add to recent logs
        self.recent_logs.append(processing_entry)
        
        # Store performance metric
        with self.metrics_lock:
            self.metrics[f"processing_time_{operation}"].append((timestamp, duration_ms))
        
        # Log to file
        self._write_log_to_file(processing_entry, "general")
            
        # Log to general logger
        self.logger.info(
            f"Data processing - {operation}: {input_details.get('size', 'N/A')} items in {duration_ms:.2f}ms"
        )
    
    def log_api_request(self,
                        api_name: str,
                        request_details: Dict[str, Any],
                        response_summary: Dict[str, Any],
                        status_code: int,
                        duration_ms: float):
        """
        Logs API requests and responses.
        
        Args:
            api_name: Name of the API called
            request_details: Details about the request
            response_summary: Summary of the response
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
        """
        timestamp = datetime.now()
        
        api_entry = {
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "api_name": api_name,
            "request_details": request_details,
            "response_summary": response_summary,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "level": LogLevel.INFO.value,
            "category": LogCategory.API.value
        }
        
        # Add to recent logs
        self.recent_logs.append(api_entry)
        
        # Store performance metric
        with self.metrics_lock:
            self.metrics[f"api_response_time_{api_name}"].append((timestamp, duration_ms))
        
        # Determine log level based on status code
        log_level = LogLevel.INFO
        if status_code >= 400:
            log_level = LogLevel.WARNING
        if status_code >= 500:
            log_level = LogLevel.ERROR
        
        # Log to file
        self._write_log_to_file(api_entry, "general")
            
        # Log to general logger
        self.logger.info(
            f"API {api_name}: status={status_code}, duration={duration_ms:.2f}ms"
        )
    
    def log_security_event(self,
                          event: str,
                          level: LogLevel,
                          details: Dict[str, Any],
                          user_id: Optional[str] = None):
        """
        Logs security-related events.
        
        Args:
            event: Description of the security event
            level: Severity level
            details: Event details
            user_id: Optional user ID
        """
        if user_id:
            self.user_id = user_id
            
        timestamp = datetime.now()
        
        security_entry = {
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "event": event,
            "level": level.value,
            "category": LogCategory.SECURITY.value,
            "details": details
        }
        
        # Add to recent logs
        self.recent_logs.append(security_entry)
        
        # Log to file (security events always go to error log for visibility)
        self._write_log_to_file(security_entry, "error")
            
        # Log to general logger
        log_method = getattr(self.logger, level.value.lower(), self.logger.warning)
        log_method(f"Security event: {event}")
    
    def log_file_operation(self,
                          operation: str,
                          file_path: str,
                          status: str,
                          details: Optional[Dict[str, Any]] = None):
        """
        Logs file operations.
        
        Args:
            operation: Type of operation (read, write, delete, etc.)
            file_path: Path to the file
            status: Operation status (success, failure)
            details: Additional operation details
        """
        timestamp = datetime.now()
        
        file_entry = {
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            "operation": operation,
            "file_path": file_path,
            "status": status,
            "details": details or {},
            "level": LogLevel.INFO.value if status == "success" else LogLevel.WARNING.value,
            "category": LogCategory.FILE_OPERATION.value
        }
        
        # Add to recent logs
        self.recent_logs.append(file_entry)
        
        # Log to file
        self._write_log_to_file(file_entry, "general")
            
        # Log to general logger
        log_level = logging.INFO if status == "success" else logging.WARNING
        self.logger.log(log_level, f"File {operation}: {file_path} - {status}")
    
    def get_recent_logs(self, 
                       count: int = 100, 
                       level: Optional[LogLevel] = None,
                       category: Optional[LogCategory] = None) -> List[Dict[str, Any]]:
        """
        Retrieves recent log entries with optional filtering.
        
        Args:
            count: Maximum number of logs to return
            level: Optional filter by log level
            category: Optional filter by category
            
        Returns:
            List of log entries
        """
        filtered_logs = list(self.recent_logs)
        
        # Apply filters
        if level:
            filtered_logs = [log for log in filtered_logs if log.get("level") == level.value]
        
        if category:
            filtered_logs = [log for log in filtered_logs if log.get("category") == category.value]
        
        # Sort by timestamp (newest first) and limit to count
        filtered_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return filtered_logs[:count]
    
    def get_model_performance_metrics(self, 
                                    model_name: Optional[str] = None,
                                    metric_name: Optional[str] = None,
                                    from_time: Optional[datetime] = None,
                                    to_time: Optional[datetime] = None) -> Dict[str, List]:
        """
        Retrieves model performance metrics with optional filtering.
        
        Args:
            model_name: Optional filter by model name
            metric_name: Optional filter by metric name
            from_time: Optional filter by start time
            to_time: Optional filter by end time
            
        Returns:
            Dictionary of metric series
        """
        with self.metrics_lock:
            filtered_metrics = {}
            
            for key, values in self.metrics.items():
                # Apply filters
                if model_name and not key.startswith(f"{model_name}_"):
                    continue
                    
                if metric_name and not key.endswith(f"_{metric_name}"):
                    continue
                
                # Filter by time range if specified
                filtered_values = values
                if from_time or to_time:
                    filtered_values = []
                    for timestamp, value in values:
                        if from_time and timestamp < from_time:
                            continue
                        if to_time and timestamp > to_time:
                            continue
                        filtered_values.append((timestamp, value))
                
                if filtered_values:
                    filtered_metrics[key] = filtered_values
            
            return filtered_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generates a summary of system performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        summary = {
            "session_start": self.start_time.isoformat(),
            "session_duration": (datetime.now() - self.start_time).total_seconds(),
            "model_performance": {},
            "api_performance": {},
            "data_processing_performance": {}
        }
        
        with self.metrics_lock:
            # Process model performance metrics
            model_metrics = {}
            for key, values in self.metrics.items():
                if key.startswith("processing_time_"):
                    category = "data_processing_performance"
                    name = key[len("processing_time_"):]
                elif key.startswith("api_response_time_"):
                    category = "api_performance"
                    name = key[len("api_response_time_"):]
                else:
                    # Assume it's a model metric
                    category = "model_performance"
                    # Split by first underscore to get model name and metric
                    parts = key.split("_", 1)
                    if len(parts) < 2:
                        continue
                    name = parts[0]
                    metric = parts[1]
                    
                    if name not in summary[category]:
                        summary[category][name] = {}
                    
                    # Calculate statistics
                    metric_values = [v for _, v in values]
                    if metric_values:
                        summary[category][name][metric] = {
                            "count": len(metric_values),
                            "mean": sum(metric_values) / len(metric_values),
                            "min": min(metric_values),
                            "max": max(metric_values)
                        }
                    continue
                
                # For processing and API metrics
                metric_values = [v for _, v in values]
                if metric_values:
                    summary[category][name] = {
                        "count": len(metric_values),
                        "mean": sum(metric_values) / len(metric_values),
                        "min": min(metric_values),
                        "max": max(metric_values),
                        "total": sum(metric_values)
                    }
        
        return summary
    
    def export_logs(self, 
                   log_type: str = "all", 
                   from_time: Optional[datetime] = None,
                   to_time: Optional[datetime] = None,
                   format: str = "json") -> Any:
        """
        Exports logs for the specified period and type.
        
        Args:
            log_type: Type of logs to export ("all", "general", "audit", "metrics", "error")
            from_time: Start time for filtering
            to_time: End time for filtering
            format: Export format ("json", "csv")
            
        Returns:
            Exported logs in the specified format
        """
        # Determine which directories to include
        log_dirs = []
        if log_type == "all" or log_type == "general":
            log_dirs.append(self.general_log_dir)
        if log_type == "all" or log_type == "audit":
            log_dirs.append(self.audit_log_dir)
        if log_type == "all" or log_type == "metrics":
            log_dirs.append(self.metrics_log_dir)
        if log_type == "all" or log_type == "error":
            log_dirs.append(self.error_log_dir)
        
        # Convert time filters to strings for comparison
        from_str = from_time.isoformat() if from_time else None
        to_str = to_time.isoformat() if to_time else None
        
        # Collect all matching log entries
        all_logs = []
        
        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                continue
                
            for filename in os.listdir(log_dir):
                if not filename.endswith('.log') and not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(log_dir, filename)
                
                # Check if file is in date range based on file name
                if from_time or to_time:
                    file_date_str = None
                    # Extract date from filename (assuming format log_YYYYMMDD.log)
                    for prefix in ["log_", "audit_", "metrics_", "error_"]:
                        if filename.startswith(prefix) and len(filename) > len(prefix) + 8:
                            date_part = filename[len(prefix):len(prefix)+8]
                            try:
                                file_date = datetime.strptime(date_part, "%Y%m%d")
                                file_date_str = file_date.strftime("%Y-%m-%d")
                                break
                            except:
                                pass
                    
                    if file_date_str:
                        if from_str and file_date_str < from_str[:10]:
                            continue
                        if to_str and file_date_str > to_str[:10]:
                            continue
                
                # Read the log file
                try:
                    with open(file_path, 'r') as f:
                        # Check file format
                        if file_path.endswith('.json'):
                            # JSON format
                            logs = json.load(f)
                            if isinstance(logs, list):
                                # Filter by time range
                                if from_str or to_str:
                                    filtered_logs = []
                                    for log in logs:
                                        timestamp = log.get("timestamp", "")
                                        if from_str and timestamp < from_str:
                                            continue
                                        if to_str and timestamp > to_str:
                                            continue
                                        filtered_logs.append(log)
                                    logs = filtered_logs
                                
                                all_logs.extend(logs)
                        else:
                            # Text log format - parse each line
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                # Parse log line (simple parsing)
                                try:
                                    # Expected format: "YYYY-MM-DD HH:MM:SS,mmm - name - level - message"
                                    parts = line.split(" - ", 3)
                                    if len(parts) >= 4:
                                        timestamp_str = parts[0].strip()
                                        # Check time range
                                        if from_str and timestamp_str < from_str:
                                            continue
                                        if to_str and timestamp_str > to_str:
                                            continue
                                            
                                        all_logs.append({
                                            "timestamp": timestamp_str,
                                            "logger": parts[1].strip(),
                                            "level": parts[2].strip(),
                                            "message": parts[3].strip(),
                                            "source": "text_log"
                                        })
                                except:
                                    # If parsing fails, add as raw message
                                    all_logs.append({
                                        "timestamp": datetime.now().isoformat(),
                                        "message": line,
                                        "source": "unparsed"
                                    })
                except Exception as e:
                    # Log the error but continue
                    self.logger.warning(f"Error reading log file {file_path}: {e}")
        
        # Sort logs by timestamp
        all_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Export in the requested format
        if format.lower() == "json":
            return json.dumps(all_logs, indent=2)
        elif format.lower() == "csv":
            # Convert to DataFrame for CSV export
            try:
                df = pd.DataFrame(all_logs)
                return df.to_csv(index=False)
            except Exception as e:
                self.logger.error(f"Error converting logs to CSV: {e}")
                return json.dumps(all_logs)  # Fall back to JSON
        else:
            # Unsupported format
            raise ValueError(f"Unsupported export format: {format}")
    
    def generate_compliance_report(self, 
                                  report_type: str = "general",
                                  from_time: Optional[datetime] = None,
                                  to_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generates compliance reports.
        
        Args:
            report_type: Type of report ("general", "user_activity", "data_access", "model_usage")
            from_time: Start time for the report
            to_time: End time for the report
            
        Returns:
            Dictionary with report data
        """
        # Set default time range if not specified
        if not to_time:
            to_time = datetime.now()
        if not from_time:
            # Default to 30 days
            from_time = to_time - timedelta(days=30)
        
        # Create report structure
        report = {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "period": {
                "from": from_time.isoformat(),
                "to": to_time.isoformat()
            },
            "system_info": self.system_info,
            "summary": {},
            "details": {}
        }
        
        # Convert time filters to strings
        from_str = from_time.isoformat()
        to_str = to_time.isoformat()
        
        # Get filtered logs
        filtered_logs = []
        for log in self.recent_logs:
            timestamp = log.get("timestamp", "")
            if timestamp >= from_str and timestamp <= to_str:
                filtered_logs.append(log)
        
        # Get additional logs from files for the specified period
        file_logs = json.loads(self.export_logs("all", from_time, to_time, "json"))
        
        # Combine logs, avoiding duplicates
        seen_ids = set()
        all_logs = []
        
        for log in filtered_logs:
            # Create a simple ID using timestamp and first part of the message/event
            log_id = f"{log.get('timestamp', '')}_{str(log.get('event', log.get('message', '')))[:20]}"
            if log_id not in seen_ids:
                all_logs.append(log)
                seen_ids.add(log_id)
        
        for log in file_logs:
            log_id = f"{log.get('timestamp', '')}_{str(log.get('event', log.get('message', '')))[:20]}"
            if log_id not in seen_ids:
                all_logs.append(log)
                seen_ids.add(log_id)
        
        # Generate report based on type
        if report_type == "general":
            # General system activity report
            
            # Count logs by level
            level_counts = defaultdict(int)
            category_counts = defaultdict(int)
            user_actions = defaultdict(int)
            error_counts = defaultdict(int)
            
            for log in all_logs:
                level = log.get("level", "UNKNOWN")
                level_counts[level] += 1
                
                category = log.get("category", "UNKNOWN")
                category_counts[category] += 1
                
                # Track user actions
                if category == LogCategory.USER_ACTION.value:
                    action = log.get("action", "UNKNOWN")
                    user_actions[action] += 1
                
                # Track errors
                if level in [LogLevel.ERROR.value, LogLevel.CRITICAL.value]:
                    error_type = "UNKNOWN"
                    if "details" in log and "exception_type" in log["details"]:
                        error_type = log["details"]["exception_type"]
                    error_counts[error_type] += 1
            
            # Build summary
            report["summary"] = {
                "total_logs": len(all_logs),
                "level_counts": dict(level_counts),
                "category_counts": dict(category_counts),
                "user_action_counts": dict(user_actions),
                "error_counts": dict(error_counts),
                "period_days": (to_time - from_time).days
            }
            
            # Add performance metrics
            performance_summary = self.get_performance_summary()
            report["summary"]["performance"] = performance_summary
            
        elif report_type == "user_activity":
            # User activity report
            
            # Filter to user actions
            user_logs = [
                log for log in all_logs 
                if log.get("category") == LogCategory.USER_ACTION.value
                or log.get("level") == LogLevel.USER.value
            ]
            
            # Group by user
            user_activity = defaultdict(list)
            for log in user_logs:
                user_id = log.get("user_id", "anonymous")
                user_activity[user_id].append(log)
            
            # Build summary
            report["summary"] = {
                "total_users": len(user_activity),
                "total_actions": len(user_logs),
                "actions_per_user": {
                    user_id: len(actions) for user_id, actions in user_activity.items()
                }
            }
            
            # Add details
            report["details"]["user_activity"] = {
                user_id: [
                    {k: v for k, v in action.items() if k not in ["details"]}
                    for action in actions
                ]
                for user_id, actions in user_activity.items()
            }
            
        elif report_type == "model_usage":
            # Model usage report
            
            # Filter to model-related logs
            model_logs = [
                log for log in all_logs 
                if log.get("category") == LogCategory.MODEL.value
                or log.get("level") == LogLevel.METRIC.value
            ]
            
            # Group by model
            model_usage = defaultdict(list)
            for log in model_logs:
                model_name = log.get("model_name", "unknown")
                model_usage[model_name].append(log)
            
            # Extract metrics
            model_metrics = {}
            for model_name, logs in model_usage.items():
                metrics = defaultdict(list)
                for log in logs:
                    if "metrics" in log:
                        for metric_name, value in log["metrics"].items():
                            if isinstance(value, (int, float)):
                                metrics[metric_name].append(value)
                
                # Calculate statistics
                model_metrics[model_name] = {
                    metric_name: {
                        "count": len(values),
                        "mean": sum(values) / len(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0
                    }
                    for metric_name, values in metrics.items()
                }
            
            # Build summary
            report["summary"] = {
                "total_models": len(model_usage),
                "total_predictions": sum(len(logs) for logs in model_usage.values()),
                "predictions_per_model": {
                    model_name: len(logs) for model_name, logs in model_usage.items()
                },
                "model_metrics": model_metrics
            }
            
        return report
    
    def _write_log_to_file(self, log_entry: Dict[str, Any], log_type: str):
        """
        Writes a log entry to the appropriate log file.
        
        Args:
            log_entry: Log entry to write
            log_type: Type of log file ("general", "audit", "metrics", "error")
        """
        if not self.enable_file:
            return
            
        # Determine log directory
        if log_type == "audit":
            log_dir = self.audit_log_dir
        elif log_type == "metrics":
            log_dir = self.metrics_log_dir
        elif log_type == "error":
            log_dir = self.error_log_dir
        else:
            log_dir = self.general_log_dir
        
        # Determine log file path
        timestamp = datetime.now()
        log_file = os.path.join(
            log_dir,
            f"{log_type}_{timestamp.strftime('%Y%m%d')}.json"
        )
        
        try:
            # Create list of logs if file doesn't exist yet
            logs = []
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
            
            # Add new log entry
            logs.append(log_entry)
            
            # Write the updated logs back to the file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            # Use Python logger directly to avoid recursion
            self.logger.error(f"Error writing to log file {log_file}: {e}")


# Create performance monitoring decorators
def log_performance(category=None, operation=None):
    """
    Decorator to log performance metrics for a function.
    
    Args:
        category: Operation category
        operation: Operation name (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger instance
            logger = LoggingSystem()
            
            # Get operation name
            op_name = operation or func.__name__
            cat_name = category or LogCategory.PERFORMANCE.value
            
            # Record start time
            start_time = time.perf_counter()
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Record end time
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Log performance
                if cat_name == LogCategory.DATA_PROCESSING.value:
                    # Get input/output details
                    input_details = {"args_count": len(args)}
                    
                    # Try to get data size if applicable
                    if args and hasattr(args[0], "__len__"):
                        input_details["size"] = len(args[0])
                    
                    output_details = {}
                    if result is not None:
                        output_details["type"] = type(result).__name__
                        if hasattr(result, "__len__"):
                            output_details["size"] = len(result)
                    
                    logger.log_data_processing(
                        op_name,
                        input_details,
                        output_details,
                        duration_ms
                    )
                else:
                    # General performance logging
                    logger.log_system_event(
                        f"Performance: {op_name}",
                        LogLevel.PERFORMANCE,
                        LogCategory.PERFORMANCE,
                        details={
                            "duration_ms": duration_ms,
                            "args_count": len(args),
                            "kwargs_count": len(kwargs)
                        }
                    )
                
                return result
                
            except Exception as e:
                # Record error time
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Log error with performance info
                logger.log_system_event(
                    f"Error in {op_name}",
                    LogLevel.ERROR,
                    LogCategory.PERFORMANCE,
                    details={
                        "duration_ms": duration_ms,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    },
                    exception=e
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator


def log_user_action(action_name):
    """
    Decorator to log user actions.
    
    Args:
        action_name: Name of the action
        
    Returns:
        Decorated function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger instance
            logger = LoggingSystem()
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Extract details
                details = {
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                
                # Log user action
                logger.log_user_action(action_name, details)
                
                return result
                
            except Exception as e:
                # Log error
                logger.log_system_event(
                    f"Error in user action: {action_name}",
                    LogLevel.ERROR,
                    LogCategory.USER_ACTION,
                    details={
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    },
                    exception=e
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator


def log_model_prediction(model_name):
    """
    Decorator to log model predictions and performance.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Decorated function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger instance
            logger = LoggingSystem()
            
            # Record start time
            start_time = time.perf_counter()
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Record end time
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Extract metrics
                metrics = {
                    "duration_ms": duration_ms
                }
                
                # Try to extract confidence scores from result
                if isinstance(result, dict):
                    if "sentiment" in result and isinstance(result["sentiment"], dict):
                        sentiment = result["sentiment"]
                        if "score" in sentiment:
                            metrics["sentiment_confidence"] = sentiment["score"]
                    
                    if "emotion" in result and isinstance(result["emotion"], dict):
                        emotion = result["emotion"]
                        if "score" in emotion:
                            metrics["emotion_confidence"] = emotion["score"]
                
                # Create prediction details
                prediction_details = {
                    "input_type": type(args[0]).__name__ if args else None,
                    "output_type": type(result).__name__
                }
                
                # Log model performance
                logger.log_model_performance(model_name, metrics, prediction_details)
                
                return result
                
            except Exception as e:
                # Record error time
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                # Log error
                logger.log_system_event(
                    f"Error in model prediction: {model_name}",
                    LogLevel.ERROR,
                    LogCategory.MODEL,
                    details={
                        "duration_ms": duration_ms,
                        "input_type": type(args[0]).__name__ if args else None
                    },
                    exception=e
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator 