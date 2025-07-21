"""
File upload component for the Streamlit application.
Provides UI for uploading and analyzing files in batch mode.
"""

import streamlit as st
import pandas as pd
import io
import time
from src.models.transformer import SentimentEmotionTransformer
from src.ui.utils.display import ResultsFormatter


class FileUploadComponent:
    """
    Creates and manages file upload interface.
    """
    
    def __init__(self):
        """
        Initialize the file upload component.
        """
        self.formatter = ResultsFormatter()
        self.supported_files = ["csv", "txt"]
        self.max_file_size_mb = 10  # Maximum file size in MB
        self.model = None
    
    def get_model(self):
        """
        Get or create the transformer model instance.
        
        Returns:
            SentimentEmotionTransformer: Model instance
        """
        if self.model is None:
            self.model = SentimentEmotionTransformer()
        return self.model
    
    def render(self):
        """
        Render the file upload UI components.
        """
        st.header("Batch File Analysis")
        
        # File upload widget with improved guidance
        uploaded_file = st.file_uploader(
            "Upload a file for batch analysis",
            type=self.supported_files,
            help=f"Supported file types: {', '.join(self.supported_files)}. Maximum file size: {self.max_file_size_mb}MB."
        )
        
        # Status area for file validation
        file_status = st.empty()
        
        # Options for batch processing
        if uploaded_file is not None:
            # Validate file first
            if not self.validate_file(uploaded_file, file_status):
                return
                
            # If validation passes, show file info
            file_size_kb = uploaded_file.size / 1024
            file_status.success(f"✅ File '{uploaded_file.name}' uploaded successfully ({file_size_kb:.1f} KB)")
            
            st.subheader("Batch Processing Options")
            
            # Column selection for CSV files
            text_column = None
            if uploaded_file.name.endswith('.csv'):
                try:
                    # Preview the CSV data
                    df_preview = pd.read_csv(uploaded_file, nrows=5)
                    st.write("Preview of uploaded CSV:")
                    st.dataframe(df_preview)
                    
                    # Allow user to select the text column
                    text_column = st.selectbox(
                        "Select text column to analyze:",
                        options=df_preview.columns.tolist()
                    )
                    
                    # Reset the file pointer for later processing
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {str(e)}")
                    st.info("💡 Tip: Make sure your CSV file is properly formatted.")
                    return
            
            # Process button with progress handling
            col1, col2 = st.columns([1, 3])
            with col1:
                process_button = st.button("Process File", use_container_width=True)
            
            with col2:
                # Add processing time estimate based on file size
                if uploaded_file.size > 500000:  # If file is larger than 500KB
                    st.caption("⚠️ Large file detected. Processing may take a minute or more.")
                else:
                    st.caption("⏱️ Processing typically takes 10-30 seconds.")
            
            # Process file when button is clicked
            if process_button:
                # Create a processing status area
                status_area = st.empty()
                status_area.info("⏳ Initializing batch processing...")
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                try:
                    # Get current parameters from session state
                    params = {
                        "sentiment_threshold": st.session_state.sentiment_threshold,
                        "emotion_threshold": st.session_state.emotion_threshold,
                        "use_api_fallback": st.session_state.use_api_fallback,
                        "text_column": text_column
                    }
                    
                    # Process uploaded file with current parameters
                    with st.spinner("🔍 Processing file..."):
                        results = self.handle_file_upload(uploaded_file, params, status_area, progress_bar)
                    
                    if results:
                        # Update status
                        status_area.success("✅ File processing completed successfully!")
                        progress_bar.progress(100)
                        
                        # Store results in session state
                        st.session_state.batch_results = results
                        st.session_state.processing_complete = True
                    else:
                        status_area.error("❌ File processing failed. Please see error details above.")
                except Exception as e:
                    status_area.error(f"❌ Unexpected error: {str(e)}")
                    st.exception(e)
        else:
            # Display help message when no file is uploaded
            st.info("⬆️ Please upload a file to begin batch analysis.")
            
            # Show example file format
            with st.expander("📝 Example file formats"):
                st.markdown("""
                **CSV File Example:**
                - First row should contain column headers
                - One column should contain the text to analyze
                
                **Text File Example:**
                - Each line will be treated as a separate text for analysis
                - Empty lines will be skipped
                """)
        
        # Display batch results if processing is complete
        if hasattr(st.session_state, "processing_complete") and st.session_state.processing_complete:
            if st.session_state.batch_results:
                self.display_batch_results(st.session_state.batch_results)
    
    def validate_file(self, file, status_area=None):
        """
        Validate uploaded file format and content.
        
        Args:
            file (UploadedFile): Uploaded file to validate
            status_area (st.empty, optional): Status area for feedback
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        # Check file size (convert bytes to MB)
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            if status_area:
                status_area.error(f"❌ File too large: {file_size_mb:.1f}MB. Maximum size is {self.max_file_size_mb}MB.")
            else:
                st.error(f"❌ File too large: {file_size_mb:.1f}MB. Maximum size is {self.max_file_size_mb}MB.")
            return False
        
        # Check file type
        file_extension = file.name.split('.')[-1].lower()
        if file_extension not in self.supported_files:
            if status_area:
                status_area.error(f"❌ Unsupported file type: .{file_extension}. Supported types: {', '.join(self.supported_files)}")
            else:
                st.error(f"❌ Unsupported file type: .{file_extension}. Supported types: {', '.join(self.supported_files)}")
            return False
        
        # Check file content based on type
        try:
            file_content = file.read()
            file.seek(0)  # Reset file pointer after reading
            
            # Check if file is empty
            if len(file_content) == 0:
                if status_area:
                    status_area.error("❌ File is empty. Please upload a file with content.")
                else:
                    st.error("❌ File is empty. Please upload a file with content.")
                return False
            
            # Additional validation for CSV files
            if file_extension == 'csv':
                try:
                    # Try to read as CSV
                    df = pd.read_csv(io.BytesIO(file_content), nrows=1)
                    
                    # Check if CSV has at least one column
                    if len(df.columns) < 1:
                        if status_area:
                            status_area.error("❌ CSV file must contain at least one column.")
                        else:
                            st.error("❌ CSV file must contain at least one column.")
                        return False
                except Exception as e:
                    if status_area:
                        status_area.error(f"❌ Invalid CSV format: {str(e)}")
                    else:
                        st.error(f"❌ Invalid CSV format: {str(e)}")
                    return False
            
            # Additional validation for TXT files
            if file_extension == 'txt':
                # Check if text file has at least one non-empty line
                text_content = file_content.decode('utf-8', errors='replace').strip()
                if not text_content:
                    if status_area:
                        status_area.error("❌ Text file is empty or contains only whitespace.")
                    else:
                        st.error("❌ Text file is empty or contains only whitespace.")
                    return False
            
            return True
            
        except Exception as e:
            if status_area:
                status_area.error(f"❌ Error validating file: {str(e)}")
            else:
                st.error(f"❌ Error validating file: {str(e)}")
            return False
    
    def handle_file_upload(self, file, params, status_area=None, progress_bar=None):
        """
        Process uploaded file with current parameters.
        
        Args:
            file (UploadedFile): Uploaded file object
            params (dict): Parameters like thresholds and API fallback setting
            status_area (st.empty, optional): Empty placeholder for status updates
            progress_bar (st.progress, optional): Progress bar for visual feedback
            
        Returns:
            dict: Batch processing results
        """
        try:
            # Update status if status_area exists
            if status_area:
                status_area.info("⏳ Loading file contents...")
            
            # Update progress
            if progress_bar:
                progress_bar.progress(10)
            
            # Load the file
            texts = self.load_file(file, text_column=params.get("text_column"))
            
            # Check if any texts were found
            if not texts or len(texts) == 0:
                st.error("❌ No valid texts found in the file.")
                return None
                
            # Update progress and status
            if progress_bar:
                progress_bar.progress(30)
            if status_area:
                status_area.info(f"⏳ Processing {len(texts)} texts...")
            
            # Process batch
            batch_results = self.process_batch(texts, params, progress_bar)
            
            # Update progress
            if progress_bar:
                progress_bar.progress(90)
            
            if status_area:
                status_area.info("⏳ Finalizing results...")
            
            # Prepare results to return
            results = {
                "file_name": file.name,
                "total_texts": len(texts),
                "batch_results": batch_results,
                "parameters": params
            }
            
            # Final progress update
            if progress_bar:
                progress_bar.progress(95)
            
            return results
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Tip: Check if your file contains valid content and is properly formatted.")
            if params["use_api_fallback"] == False:
                st.info("💡 Tip: Try enabling API fallback in the sidebar for alternative processing.")
            return None
    
    def load_file(self, file, text_column=None):
        """
        Load texts from uploaded file.
        
        Args:
            file (UploadedFile): Uploaded file object
            text_column (str): Column name for CSV files
            
        Returns:
            list: List of text strings
        """
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            if text_column and text_column in df.columns:
                texts = df[text_column].dropna().tolist()
            else:
                # Use first column if no column specified
                texts = df.iloc[:, 0].dropna().tolist()
        else:
            # For txt files, read line by line
            content = file.read().decode('utf-8')
            texts = [line.strip() for line in content.split('\n') if line.strip()]
        
        return texts
    
    def process_batch(self, texts, params, progress_bar=None):
        """
        Process a batch of texts.
        
        Args:
            texts (list): List of text strings to analyze
            params (dict): Analysis parameters
            progress_bar (st.progress, optional): Progress bar for visual feedback
            
        Returns:
            dict: Batch processing results
        """
        model = self.get_model()
        model.set_thresholds(
            sentiment_threshold=params["sentiment_threshold"],
            emotion_threshold=params["emotion_threshold"]
        )
        
        results = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        emotion_distribution = {}
        total_confidence = 0
        
        # Process each text
        for i, text in enumerate(texts):
            try:
                # Update progress for each text
                if progress_bar:
                    progress = 30 + int((i / len(texts)) * 60)
                    progress_bar.progress(progress)
                
                analysis_result = model.analyze(text, use_fallback=params["use_api_fallback"])
                
                sentiment = analysis_result.get("sentiment", {})
                emotion = analysis_result.get("emotion", {})
                
                # Count sentiment
                sentiment_label = sentiment.get("label", "neutral")
                sentiment_counts[sentiment_label] += 1
                
                # Count emotions
                emotion_scores = emotion.get("scores", {})
                for emotion_name, score in emotion_scores.items():
                    emotion_distribution[emotion_name] = emotion_distribution.get(emotion_name, 0) + score
                
                # Accumulate confidence
                confidence = sentiment.get("score", 0.0)
                total_confidence += confidence
                
                # Store detailed result
                results.append({
                    "text": text,
                    "sentiment": sentiment_label,
                    "sentiment_confidence": confidence,
                    "emotions": emotion_scores,
                    "index": i
                })
                
            except Exception as e:
                st.warning(f"⚠️ Error processing text {i+1}: {str(e)}")
                results.append({
                    "text": text,
                    "sentiment": "error",
                    "sentiment_confidence": 0.0,
                    "emotions": {},
                    "index": i
                })
        
        # Calculate averages
        average_confidence = total_confidence / len(texts) if texts else 0.0
        
        # Normalize emotion distribution
        for emotion in emotion_distribution:
            emotion_distribution[emotion] /= len(texts)
        
        # Update session statistics
        from src.ui.utils.session import SessionManager
        session_manager = SessionManager()
        session_manager.increment_texts_analyzed(len(texts), is_groq=False)
        
        return {
            "detailed_results": results,
            "sentiment_counts": sentiment_counts,
            "emotion_distribution": emotion_distribution,
            "average_confidence": average_confidence
        }
    
    def display_batch_results(self, results):
        """
        Format and display batch processing results.
        
        Args:
            results (dict): Batch processing results to display
        """
        # Collapsible results section
        with st.expander("📊 Batch Processing Results", expanded=True):
            st.subheader("Summary")
            
            # Display summary
            st.markdown(f"**File:** `{results['file_name']}`")
            st.markdown(f"**Total Texts Processed:** {results['total_texts']}")
            
            # Create metrics for overall results
            batch_results = results["batch_results"]
            
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            # Sentiment metrics with emojis
            with col1:
                st.metric("🟢 Positive Texts", batch_results["sentiment_counts"].get("positive", 0))
            
            with col2:
                st.metric("🔴 Negative Texts", batch_results["sentiment_counts"].get("negative", 0))
            
            with col3:
                st.metric("⚪ Neutral Texts", batch_results["sentiment_counts"].get("neutral", 0))
            
            with col4:
                st.metric("🎯 Avg. Confidence", f"{batch_results['average_confidence']:.2%}")
            
            # Display emotion distribution
            st.markdown("### Emotion Distribution")
            self.formatter.create_bar_chart(
                data=batch_results["emotion_distribution"], 
                title="Emotion Distribution Across All Texts"
            )
            
            # Display full results table with filtering
            st.markdown("### Detailed Results")
            
            # Convert to DataFrame for filtering
            df_results = pd.DataFrame(batch_results["detailed_results"])
            
            # Add filter options
            filter_container = st.container()
            with filter_container:
                filter_col1, filter_col2 = st.columns(2)
                
                # Filter by sentiment if sentiment column exists
                if "sentiment" in df_results.columns:
                    with filter_col1:
                        sentiment_filter = st.multiselect(
                            "Filter by Sentiment",
                            options=sorted(df_results["sentiment"].unique()),
                            default=sorted(df_results["sentiment"].unique())
                        )
                
                # Filter by confidence if confidence column exists
                if "sentiment_confidence" in df_results.columns:
                    with filter_col2:
                        min_confidence = st.slider(
                            "Minimum Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.05
                        )
            
            # Apply filters
            filtered_df = df_results
            
            if "sentiment" in df_results.columns and sentiment_filter:
                filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_filter)]
            
            if "sentiment_confidence" in df_results.columns:
                filtered_df = filtered_df[filtered_df["sentiment_confidence"] >= min_confidence]
            
            # Display filtered results
            st.write(f"Showing {len(filtered_df)} of {len(df_results)} results")
            self.formatter.create_results_table(filtered_df.to_dict("records"))
            
            # Export functionality
            st.markdown("### Export Results")
            
            # Enhanced export with ExportManager
            if st.button("📥 Export Results", key="export_batch_results"):
                # Use the ExportManager
                from src.ui.components.export_manager import ExportManager
                export_manager = ExportManager()
                export_manager.render_export_panel("batch", results)
            
            # Quick download buttons for immediate access
            download_col1, download_col2 = st.columns(2)
            
            # Download all results
            with download_col1:
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "Quick Download (CSV)",
                    csv,
                    file_name=f"batch_analysis_results_{results['file_name'].split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Download filtered results
            with download_col2:
                filtered_csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Quick Download Filtered (CSV)",
                    filtered_csv,
                    file_name=f"filtered_results_{results['file_name'].split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Display the parameters used with better formatting
            st.markdown("### Parameters Used")
            params_col1, params_col2, params_col3 = st.columns(3)
            with params_col1:
                st.metric("Sentiment Threshold", f"{results['parameters']['sentiment_threshold']:.2f}")
            with params_col2:
                st.metric("Emotion Threshold", f"{results['parameters']['emotion_threshold']:.2f}")
            with params_col3:
                st.metric("API Fallback", "Enabled" if results['parameters']['use_api_fallback'] else "Disabled")
            
            # Add report generation section
            from src.ui.components.report_generator import ReportGenerator
            
            st.markdown("### Generate Report")
            report_generator = ReportGenerator()
            report_generator.render_options("standard", results) 