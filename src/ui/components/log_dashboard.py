import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Optional

from src.utils.logging_system import LoggingSystem, LogLevel, LogCategory


class LogDashboard:
    """
    Dashboard for viewing and managing system logs,
    performance metrics, and audit trails.
    """
    
    def __init__(self, session_state=None, key_manager=None):
        """
        Initialize the log dashboard.
        
        Args:
            session_state: Streamlit session state
            key_manager: Widget key manager for unique keys
        """
        self.session_state = session_state or st.session_state
        self.logger = LoggingSystem()
        self.key_manager = key_manager
        if self.key_manager:
            self.key_manager.register_component('log_dashboard', 'ld')
    
    def render(self):
        """
        Renders the log dashboard UI.
        """
        st.title("System Logs & Audit Trail")
        
        # Create tabs
        tabs = st.tabs([
            "Recent Logs",
            "Performance Metrics",
            "User Activity",
            "Model Performance",
            "Compliance Reports",
            "Export Logs"
        ])
        
        # Recent Logs tab
        with tabs[0]:
            self.render_logs_view()
        
        # Performance Metrics tab
        with tabs[1]:
            self.render_performance_metrics()
        
        # User Activity tab
        with tabs[2]:
            self.render_user_activity()
        
        # Model Performance tab
        with tabs[3]:
            self.render_model_performance()
        
        # Compliance Reports tab
        with tabs[4]:
            self.render_compliance_reports()
        
        # Export Logs tab
        with tabs[5]:
            self.render_export_logs()
    
    def render_logs_view(self):
        """Renders the recent logs view."""
        st.header("System Logs")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Log level filter
            level_options = [level.value for level in LogLevel]
            selected_level = st.selectbox(
                "Filter by Level",
                options=["All"] + level_options,
                index=0,
                key=self.key_manager.get_key('log_dashboard_logs', 'level_filter') if self.key_manager else None
            )
        
        with col2:
            # Category filter
            category_options = [category.value for category in LogCategory]
            selected_category = st.selectbox(
                "Filter by Category",
                options=["All"] + category_options,
                index=0,
                key=self.key_manager.get_key('log_dashboard_logs', 'category_filter') if self.key_manager else None
            )
        
        with col3:
            # Log count
            log_count = st.number_input(
                "Number of Logs",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key=self.key_manager.get_key('log_dashboard_logs', 'log_count') if self.key_manager else None
            )
        
        # Apply filters
        level_filter = None if selected_level == "All" else LogLevel(selected_level)
        category_filter = None if selected_category == "All" else LogCategory(selected_category)
        
        # Get logs
        logs = self.logger.get_recent_logs(log_count, level_filter, category_filter)
        
        # Display logs
        if logs:
            # Create a DataFrame for display
            df_data = []
            for log in logs:
                row = {
                    "TimeStamp": log.get("timestamp", ""),
                    "Level": log.get("level", ""),
                    "Category": log.get("category", ""),
                    "Message": log.get("event", log.get("message", "")),
                }
                df_data.append(row)
            
            log_df = pd.DataFrame(df_data)
            st.dataframe(log_df, use_container_width=True)
            
            # Create download button
            csv = log_df.to_csv(index=False)
            st.download_button("Download Logs", csv, "system_logs.csv", "text/csv")
        else:
            st.info("No logs found with the current filters.")
            
        # Add log details expander
        if logs:
            with st.expander("View Log Details"):
                # Allow selection of a specific log
                log_indices = range(len(logs))
                selected_log_index = st.selectbox(
                    "Select Log Entry",
                    options=log_indices,
                    format_func=lambda i: f"{logs[i].get('timestamp', '')} - {logs[i].get('level', '')} - {logs[i].get('event', logs[i].get('message', ''))[:50]}"
                )
                
                # Display selected log
                selected_log = logs[selected_log_index]
                st.json(selected_log)
    
    def render_performance_metrics(self):
        """Renders performance metrics charts."""
        st.header("Performance Metrics")
        
        # Get summary
        summary = self.logger.get_performance_summary()
        
        # Display session info
        st.subheader("Session Information")
        
        # Calculate session duration
        if 'session_start' in summary:
            try:
                start_time = datetime.fromisoformat(summary['session_start'])
                duration_seconds = summary['session_duration']
                
                # Format duration
                duration_str = str(timedelta(seconds=int(duration_seconds)))
                
                # Display info
                st.text(f"Session started: {start_time}")
                st.text(f"Session duration: {duration_str}")
            except:
                st.text("Session information not available")
        
        # Model performance
        st.subheader("Model Performance")
        
        model_metrics = summary.get('model_performance', {})
        if model_metrics:
            # Create tabs for each model
            model_tabs = st.tabs(list(model_metrics.keys()))
            
            for i, model_name in enumerate(model_metrics.keys()):
                with model_tabs[i]:
                    model_data = model_metrics[model_name]
                    
                    # Create metrics display
                    metric_cols = st.columns(len(model_data))
                    
                    for j, (metric_name, stats) in enumerate(model_data.items()):
                        with metric_cols[j]:
                            st.metric(
                                label=metric_name,
                                value=f"{stats['mean']:.3f}",
                                delta=None
                            )
                            st.text(f"Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")
                            st.text(f"Count: {stats['count']}")
        else:
            st.info("No model performance metrics available.")
        
        # API performance
        st.subheader("API Performance")
        
        api_metrics = summary.get('api_performance', {})
        if api_metrics:
            # Create a table
            api_data = []
            for api_name, stats in api_metrics.items():
                api_data.append({
                    "API": api_name,
                    "Avg. Response Time (ms)": f"{stats['mean']:.2f}",
                    "Min (ms)": f"{stats['min']:.2f}",
                    "Max (ms)": f"{stats['max']:.2f}",
                    "Count": stats['count']
                })
            
            api_df = pd.DataFrame(api_data)
            st.dataframe(api_df, use_container_width=True)
        else:
            st.info("No API performance metrics available.")
        
        # Data processing performance
        st.subheader("Data Processing Performance")
        
        processing_metrics = summary.get('data_processing_performance', {})
        if processing_metrics:
            # Create a table
            proc_data = []
            for op_name, stats in processing_metrics.items():
                proc_data.append({
                    "Operation": op_name,
                    "Avg. Duration (ms)": f"{stats['mean']:.2f}",
                    "Min (ms)": f"{stats['min']:.2f}",
                    "Max (ms)": f"{stats['max']:.2f}",
                    "Count": stats['count'],
                    "Total (ms)": f"{stats['total']:.2f}"
                })
            
            proc_df = pd.DataFrame(proc_data)
            st.dataframe(proc_df, use_container_width=True)
        else:
            st.info("No data processing performance metrics available.")
        
        # Performance charts
        st.subheader("Performance Trends")
        
        # Get raw metrics data
        metrics_data = self.logger.get_model_performance_metrics()
        
        if metrics_data:
            # Allow metric selection
            metric_options = list(metrics_data.keys())
            selected_metric = st.selectbox(
                "Select Metric",
                options=metric_options
            )
            
            # Get selected metric data
            metric_series = metrics_data.get(selected_metric, [])
            
            if metric_series:
                # Create DataFrame
                df = pd.DataFrame(metric_series, columns=["timestamp", "value"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
                
                # Plot
                st.line_chart(df)
            else:
                st.info("No data available for selected metric.")
        else:
            st.info("No performance metrics data available.")
    
    def render_user_activity(self):
        """Renders user activity logs."""
        st.header("User Activity")
        
        # Get user actions
        user_logs = self.logger.get_recent_logs(
            1000, 
            LogLevel.AUDIT, 
            LogCategory.USER_ACTION
        )
        
        if user_logs:
            # Group by user
            user_activity = defaultdict(list)
            for log in user_logs:
                user_id = log.get("user_id", "anonymous")
                user_activity[user_id].append(log)
            
            # Display summary
            st.subheader("Activity Summary")
            
            summary_data = []
            for user_id, logs in user_activity.items():
                summary_data.append({
                    "User": user_id,
                    "Action Count": len(logs),
                    "Last Activity": max(log.get("timestamp", "") for log in logs)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values("Last Activity", ascending=False)
            st.dataframe(summary_df, use_container_width=True)
            
            # Activity timeline
            st.subheader("Activity Timeline")
            
            # Create timeline data
            timeline_data = []
            for log in user_logs:
                timeline_data.append({
                    "Timestamp": pd.to_datetime(log.get("timestamp", "")),
                    "User": log.get("user_id", "anonymous"),
                    "Action": log.get("action", "Unknown")
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values("Timestamp", ascending=False)
            
            st.dataframe(timeline_df, use_container_width=True)
        else:
            st.info("No user activity logs available.")
    
    def render_model_performance(self):
        """Renders model performance analysis."""
        st.header("Model Performance Analysis")
        
        # Get model metrics
        model_logs = self.logger.get_recent_logs(
            1000, 
            LogLevel.METRIC, 
            LogCategory.MODEL
        )
        
        if model_logs:
            # Group by model
            model_metrics = defaultdict(list)
            for log in model_logs:
                model_name = log.get("model_name", "unknown")
                model_metrics[model_name].append(log)
            
            # Display summary
            st.subheader("Model Summary")
            
            summary_data = []
            for model_name, logs in model_metrics.items():
                # Extract average metrics
                metrics_sum = defaultdict(float)
                metrics_count = defaultdict(int)
                
                for log in logs:
                    if "metrics" in log:
                        for metric_name, value in log["metrics"].items():
                            if isinstance(value, (int, float)):
                                metrics_sum[metric_name] += value
                                metrics_count[metric_name] += 1
                
                # Calculate averages
                avg_metrics = {}
                for metric_name, total in metrics_sum.items():
                    if metrics_count[metric_name] > 0:
                        avg_metrics[metric_name] = total / metrics_count[metric_name]
                
                # Create summary row
                summary_row = {
                    "Model": model_name,
                    "Prediction Count": len(logs)
                }
                
                # Add metrics
                for metric_name, avg_value in avg_metrics.items():
                    summary_row[f"Avg {metric_name}"] = avg_value
                
                summary_data.append(summary_row)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Detailed metrics
            st.subheader("Metrics Detail")
            
            # Allow model selection
            model_options = list(model_metrics.keys())
            selected_model = st.selectbox(
                "Select Model",
                options=model_options
            )
            
            # Get model logs
            model_logs = model_metrics.get(selected_model, [])
            
            if model_logs:
                # Extract all metrics
                all_metrics = set()
                for log in model_logs:
                    if "metrics" in log:
                        all_metrics.update(log["metrics"].keys())
                
                # Allow metric selection
                metric_options = list(all_metrics)
                selected_metric = st.selectbox(
                    "Select Metric",
                    options=metric_options
                )
                
                # Create metric time series
                metric_data = []
                for log in model_logs:
                    if "metrics" in log and selected_metric in log["metrics"]:
                        value = log["metrics"][selected_metric]
                        if isinstance(value, (int, float)):
                            metric_data.append({
                                "timestamp": pd.to_datetime(log.get("timestamp", "")),
                                "value": value
                            })
                
                if metric_data:
                    metric_df = pd.DataFrame(metric_data)
                    metric_df = metric_df.set_index("timestamp")
                    metric_df = metric_df.sort_index()
                    
                    # Plot
                    st.line_chart(metric_df)
                    
                    # Statistics
                    st.subheader("Metric Statistics")
                    
                    stats = metric_df["value"].describe()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean", f"{stats['mean']:.3f}")
                    
                    with col2:
                        st.metric("Median", f"{stats['50%']:.3f}")
                    
                    with col3:
                        st.metric("Min", f"{stats['min']:.3f}")
                    
                    with col4:
                        st.metric("Max", f"{stats['max']:.3f}")
                else:
                    st.info("No data available for selected metric.")
            else:
                st.info("No logs available for selected model.")
        else:
            st.info("No model performance logs available.")
    
    def render_compliance_reports(self):
        """Renders compliance reports."""
        st.header("Compliance Reports")
        
        # Report type selection
        report_type = st.selectbox(
            "Report Type",
            options=["General System Activity", "User Activity", "Model Usage"],
            index=0
        )
        
        # Time range selection
        col1, col2 = st.columns(2)
        
        with col1:
            from_date = st.date_input(
                "From Date",
                value=datetime.now() - timedelta(days=30),
                key=self.key_manager.get_key('log_dashboard_compliance', 'from_date') if self.key_manager else None
            )
        
        with col2:
            to_date = st.date_input(
                "To Date",
                value=datetime.now(),
                key=self.key_manager.get_key('log_dashboard_compliance', 'to_date') if self.key_manager else None
            )
        
        # Convert to datetime objects
        from_datetime = datetime.combine(from_date, datetime.min.time())
        to_datetime = datetime.combine(to_date, datetime.max.time())
        
        # Map UI option to report type
        report_type_map = {
            "General System Activity": "general",
            "User Activity": "user_activity",
            "Model Usage": "model_usage"
        }
        
        # Generate report button
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                # Generate the report
                report = self.logger.generate_compliance_report(
                    report_type_map[report_type],
                    from_datetime,
                    to_datetime
                )
                
                # Display report
                st.json(report)
                
                # Create download button
                report_json = json.dumps(report, indent=2)
                filename = f"compliance_report_{report_type_map[report_type]}_{datetime.now().strftime('%Y%m%d')}.json"
                
                st.download_button(
                    "Download Report",
                    report_json,
                    filename,
                    "application/json"
                )
    
    def render_export_logs(self):
        """Renders log export options."""
        st.header("Export Logs")
        
        # Log type selection
        log_type = st.selectbox(
            "Log Type",
            options=["All Logs", "General Logs", "Audit Logs", "Metrics", "Error Logs"],
            index=0
        )
        
        # Time range selection
        col1, col2 = st.columns(2)
        
        with col1:
            from_date = st.date_input(
                "From Date",
                value=datetime.now() - timedelta(days=7),
                key=self.key_manager.get_key('log_dashboard_export', 'from_date') if self.key_manager else None
            )
        
        with col2:
            to_date = st.date_input(
                "To Date",
                value=datetime.now(),
                key=self.key_manager.get_key('log_dashboard_export', 'to_date') if self.key_manager else None
            )
        
        # Export format selection
        export_format = st.selectbox(
            "Export Format",
            options=["JSON", "CSV"],
            index=0
        )
        
        # Convert to datetime objects
        from_datetime = datetime.combine(from_date, datetime.min.time())
        to_datetime = datetime.combine(to_date, datetime.max.time())
        
        # Map UI option to log type
        log_type_map = {
            "All Logs": "all",
            "General Logs": "general",
            "Audit Logs": "audit",
            "Metrics": "metrics",
            "Error Logs": "error"
        }
        
        # Export button
        if st.button("Export Logs"):
            with st.spinner("Exporting logs..."):
                # Export logs
                log_data = self.logger.export_logs(
                    log_type_map[log_type],
                    from_datetime,
                    to_datetime,
                    export_format.lower()
                )
                
                # Create download button
                extension = export_format.lower()
                mime_type = "application/json" if extension == "json" else "text/csv"
                
                filename = f"logs_{log_type_map[log_type]}_{datetime.now().strftime('%Y%m%d')}.{extension}"
                
                st.download_button(
                    "Download Logs",
                    log_data,
                    filename,
                    mime_type
                ) 