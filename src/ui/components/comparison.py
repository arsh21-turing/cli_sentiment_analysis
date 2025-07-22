"""
Model comparison UI components for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from src.ui.utils.comparison_utils import ModelComparator
from src.ui.utils.display import ResultsFormatter
from src.ui.components.report_generator import ReportGenerator
import plotly.graph_objects as go
import plotly.express as px


class ComparisonComponent:
    """
    Creates and manages the comparison interface.
    """
    
    def __init__(self):
        """
        Initialize comparison component.
        """
        self.comparator = ModelComparator()
        self.formatter = ResultsFormatter()
        self.report_generator = ReportGenerator()
    
    def render(self):
        """
        Render the comparison UI components.
        """
        st.header("Model Comparison")
        
        # Check if there are at least two models available
        available_models = self.comparator.get_available_models()
        if len(available_models) < 2:
            st.warning("⚠️ Model comparison requires at least two different models. Enable Groq API in the sidebar to compare with Groq models.")
            
            # Show explanation
            st.markdown("""
            ## Why Compare Models?
            
            Comparing different models can provide:
            - More reliable results through consensus
            - Insights into model strengths and weaknesses
            - Better understanding of confidence levels
            - Identification of edge cases
            
            Enable Groq API in the sidebar settings to access this feature.
            """)
            return
        
        # Create tabs for different comparison modes
        tab1, tab2 = st.tabs(["Single Text Comparison", "Batch Comparison"])
        
        # Single text comparison tab
        with tab1:
            self.render_single_comparison(available_models)
        
        # Batch comparison tab
        with tab2:
            self.render_batch_comparison(available_models)
    
    def render_single_comparison(self, available_models: List[str]):
        """
        Render UI for comparing single text analysis.
        
        Args:
            available_models (List[str]): List of available model identifiers
        """
        st.subheader("Compare Models on Single Text")
        
        # Text input area
        text = st.text_area(
            "Enter text to analyze with multiple models:",
            height=150,
            placeholder="Type or paste your text here for comparison across models..."
        )
        
        # Model selection
        st.markdown("### Select Models to Compare")
        
        # Convert available_models to display names for better readability
        model_names = {model_id: self.comparator.model_names.get(model_id, model_id) 
                      for model_id in available_models}
        
        # Create columns for model selection
        model_cols = st.columns(min(len(available_models), 3))
        selected_models = []
        
        for i, model_id in enumerate(available_models):
            col_idx = i % len(model_cols)
            with model_cols[col_idx]:
                if st.checkbox(model_names[model_id], value=(i < 2), key=f"model_{model_id}"):
                    selected_models.append(model_id)
        
        # Analysis options
        st.markdown("### Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analyze_sentiment = st.checkbox("Analyze Sentiment", value=True, key="compare_sentiment")
        
        with col2:
            analyze_emotion = st.checkbox("Analyze Emotions", value=True, key="compare_emotion")
        
        # Determine analysis type
        analysis_type = "both"
        if analyze_sentiment and not analyze_emotion:
            analysis_type = "sentiment"
        elif not analyze_sentiment and analyze_emotion:
            analysis_type = "emotion"
        
        # Compare button
        if st.button("Compare Models", disabled=not text or len(selected_models) < 2 or not (analyze_sentiment or analyze_emotion)):
            if not text:
                st.error("❌ Please enter text to analyze.")
            elif len(selected_models) < 2:
                st.error("❌ Please select at least two models to compare.")
            elif not (analyze_sentiment or analyze_emotion):
                st.error("❌ Please select at least one analysis type.")
            else:
                with st.spinner("🔄 Comparing models..."):
                    # Perform comparison
                    results = self.compare_models(text, selected_models, analysis_type)
                    
                    # Store results in session state
                    st.session_state.comparison_results = results
                    st.session_state.comparison_complete = True
        
        # Display results if available
        if hasattr(st.session_state, "comparison_complete") and st.session_state.comparison_complete:
            if hasattr(st.session_state, "comparison_results") and st.session_state.comparison_results:
                self.display_comparison_results(st.session_state.comparison_results, "single")
    
    def render_batch_comparison(self, available_models: List[str]):
        """
        Render UI for comparing batch analysis.
        
        Args:
            available_models (List[str]): List of available model identifiers
        """
        st.subheader("Compare Models on Multiple Texts")
        
        # File upload or text area input
        input_method = st.radio(
            "Input Method",
            options=["Upload File", "Enter Text List"],
            horizontal=True
        )
        
        texts = []
        
        if input_method == "Upload File":
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload a file for batch comparison",
                type=["csv", "txt"],
                help="CSV files should have a text column. TXT files should have one text per line."
            )
            
            if uploaded_file:
                # Process the file
                try:
                    if uploaded_file.name.endswith('.csv'):
                        # Preview the CSV
                        df = pd.read_csv(uploaded_file)
                        st.write("Preview:")
                        st.dataframe(df.head(3))
                        
                        # Select text column
                        text_column = st.selectbox(
                            "Select text column:",
                            options=df.columns.tolist()
                        )
                        
                        # Get texts
                        texts = df[text_column].dropna().tolist()
                        
                        # Show count
                        st.info(f"Found {len(texts)} texts in column '{text_column}'")
                        
                    else:  # txt file
                        # Read the file
                        text_content = uploaded_file.getvalue().decode("utf-8")
                        
                        # Split into lines
                        texts = [line.strip() for line in text_content.split('\n') if line.strip()]
                        
                        # Show count
                        st.info(f"Found {len(texts)} non-empty lines in file")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.info("💡 Tip: Make sure your file is properly formatted.")
                    texts = []
        else:
            # Text area input
            text_list = st.text_area(
                "Enter multiple texts (one per line):",
                height=200,
                placeholder="Enter one text per line for batch comparison..."
            )
            
            if text_list:
                # Split into lines
                texts = [line.strip() for line in text_list.split('\n') if line.strip()]
                
                # Show count
                st.info(f"Found {len(texts)} non-empty texts")
        
        # Limit batch size
        max_batch_size = 100
        if len(texts) > max_batch_size:
            st.warning(f"⚠️ Large batch detected. Limiting to first {max_batch_size} texts for performance.")
            texts = texts[:max_batch_size]
        
        # Model selection
        if texts:
            st.markdown("### Select Models to Compare")
            
            # Convert available_models to display names for better readability
            model_names = {model_id: self.comparator.model_names.get(model_id, model_id) 
                          for model_id in available_models}
            
            # Create columns for model selection
            model_cols = st.columns(min(len(available_models), 3))
            selected_models = []
            
            for i, model_id in enumerate(available_models):
                col_idx = i % len(model_cols)
                with model_cols[col_idx]:
                    if st.checkbox(model_names[model_id], value=(i < 2), key=f"batch_model_{model_id}"):
                        selected_models.append(model_id)
            
            # Analysis options
            st.markdown("### Analysis Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                analyze_sentiment = st.checkbox("Analyze Sentiment", value=True, key="batch_sentiment")
            
            with col2:
                analyze_emotion = st.checkbox("Analyze Emotions", value=False, key="batch_emotion")
            
            # Determine analysis type
            analysis_type = "both"
            if analyze_sentiment and not analyze_emotion:
                analysis_type = "sentiment"
            elif not analyze_sentiment and analyze_emotion:
                analysis_type = "emotion"
            
            # Batch size slider
            batch_size = st.slider(
                "Batch Size (per model)",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of texts to process in parallel per model"
            )
            
            # Compare button
            if st.button("Compare Models on Batch", disabled=len(texts) == 0 or len(selected_models) < 2 or not (analyze_sentiment or analyze_emotion)):
                if len(texts) == 0:
                    st.error("❌ No texts found to analyze.")
                elif len(selected_models) < 2:
                    st.error("❌ Please select at least two models to compare.")
                elif not (analyze_sentiment or analyze_emotion):
                    st.error("❌ Please select at least one analysis type.")
                else:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_area = st.empty()
                    status_area.info("⏳ Initializing batch comparison...")
                    
                    try:
                        # Perform batch comparison
                        with st.spinner("🔄 Comparing models on batch..."):
                            results = self.compare_batches(
                                texts, 
                                selected_models, 
                                analysis_type,
                                batch_size,
                                lambda p: progress_bar.progress(int(p * 100)),
                                status_area
                            )
                        
                        # Update status
                        status_area.success("✅ Batch comparison completed successfully!")
                        progress_bar.progress(100)
                        
                        # Store results in session state
                        st.session_state.batch_comparison_results = results
                        st.session_state.batch_comparison_complete = True
                    
                    except Exception as e:
                        status_area.error(f"❌ Error in batch comparison: {str(e)}")
                        st.exception(e)
        
        # Display results if available
        if hasattr(st.session_state, "batch_comparison_complete") and st.session_state.batch_comparison_complete:
            if hasattr(st.session_state, "batch_comparison_results") and st.session_state.batch_comparison_results:
                self.display_comparison_results(st.session_state.batch_comparison_results, "batch")
    
    def compare_models(self, text: str, models: List[str], analysis_type: str = "both") -> Dict[str, Any]:
        """
        Compare multiple models on the same text.
        
        Args:
            text (str): Text to analyze with multiple models
            models (List[str]): List of models to compare
            analysis_type (str): Type of analysis
            
        Returns:
            Dict: Comparison results
        """
        # Use the ModelComparator to perform the comparison
        results = self.comparator.analyze_with_models(text, models, analysis_type)
        
        # Update session stats
        from src.ui.utils.session import SessionManager
        session_manager = SessionManager()
        session_manager.increment_texts_analyzed(1, is_groq=any(model.startswith("groq_") for model in models))
        
        return results
    
    def compare_batches(self, texts: List[str], models: List[str], analysis_type: str = "both", 
                       batch_size: int = 3, progress_callback: Optional[callable] = None,
                       status_area: Optional[Any] = None) -> Dict[str, Any]:
        """
        Compare multiple models on a batch of texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            models (List[str]): List of models to compare
            analysis_type (str): Type of analysis
            batch_size (int): Batch size for parallel processing
            progress_callback (callable, optional): Callback for progress updates
            status_area (st.empty, optional): Status area for updates
            
        Returns:
            Dict: Batch comparison results
        """
        # Update status
        if status_area:
            status_area.info(f"⏳ Comparing {len(models)} models on {len(texts)} texts...")
        
        # Use the ModelComparator to perform the batch comparison
        results = self.comparator.process_batch_comparison(
            texts, 
            models, 
            analysis_type,
            progress_callback
        )
        
        # Update session stats
        from src.ui.utils.session import SessionManager
        session_manager = SessionManager()
        session_manager.increment_texts_analyzed(
            len(texts), 
            is_groq=any(model.startswith("groq_") for model in models)
        )
        
        return results
    
    def display_comparison_results(self, results: Dict[str, Any], mode: str = "single"):
        """
        Display comparison results.
        
        Args:
            results (Dict): Comparison results to display
            mode (str): Display mode ("single" or "batch")
        """
        if mode == "single":
            self._display_single_comparison_results(results)
        else:
            self._display_batch_comparison_results(results)
    
    def _display_single_comparison_results(self, results: Dict[str, Any]):
        """
        Display results for single text comparison.
        
        Args:
            results (Dict): Single text comparison results
        """
        st.subheader("Comparison Results")
        
        # Display the analyzed text
        with st.expander("Analyzed Text", expanded=True):
            st.text_area("Text", value=results.get("text", ""), height=100, disabled=True)
        
        # Get the model IDs
        model_ids = [model_id for model_id in results.get("models_compared", [])]
        
        # Create tabs for different result views
        result_tab1, result_tab2, result_tab3 = st.tabs([
            "Side-by-Side Comparison", 
            "Comparison Metrics", 
            "Processing Statistics"
        ])
        
        with result_tab1:
            self._display_side_by_side_results(results, model_ids)
        
        with result_tab2:
            self._display_comparison_metrics(results)
        
        with result_tab3:
            self._display_processing_stats(results, model_ids)
        
        # Export functionality
        st.markdown("---")
        if st.button("📥 Export Comparison Results", key="export_comparison_results"):
            # Use the ExportManager
            from src.ui.components.export_manager import ExportManager
            export_manager = ExportManager()
            export_manager.render_export_panel("comparison", results)
        
        # Generate comparison report
        st.markdown("### Generate Comparison Report")
        self.report_generator.render_options("comparison", results)
    
    def _display_batch_comparison_results(self, results: Dict[str, Any]):
        """
        Display results for batch comparison.
        
        Args:
            results (Dict): Batch comparison results
        """
        st.subheader("Batch Comparison Results")
        
        # Display summary
        st.markdown(f"**Total Texts Analyzed:** {results.get('total_texts', 0)}")
        st.markdown(f"**Models Compared:** {', '.join(results.get('model_names', []))}")
        
        # Create tabs for different result views
        result_tab1, result_tab2, result_tab3 = st.tabs([
            "Agreement Analysis", 
            "Model Performance", 
            "Detailed Results"
        ])
        
        with result_tab1:
            self._display_agreement_analysis(results)
        
        with result_tab2:
            self._display_model_performance(results)
        
        with result_tab3:
            self._display_detailed_batch_results(results)
        
        # Export functionality
        st.markdown("---")
        if st.button("📥 Export Batch Comparison", key="export_batch_comparison"):
            # Use the ExportManager
            from src.ui.components.export_manager import ExportManager
            export_manager = ExportManager()
            export_manager.render_export_panel("comparison_batch", results)
        
        # Generate comparison report
        st.markdown("### Generate Comparison Report")
        self.report_generator.render_options("batch_comparison", results)
    
    def _display_side_by_side_results(self, results: Dict[str, Any], model_ids: List[str]):
        """
        Display side-by-side comparison of model results.
        
        Args:
            results (Dict): Comparison results
            model_ids (List[str]): List of model IDs
        """
        st.markdown("### Side-by-Side Model Comparison")
        
        # Create columns for each model
        model_columns = st.columns(len(model_ids))
        
        # Display results for each model
        for i, model_id in enumerate(model_ids):
            with model_columns[i]:
                model_result = results.get(model_id, {})
                model_name = model_result.get("model_name", model_id)
                
                st.markdown(f"#### {model_name}")
                
                # Check if there was an error
                if "error" in model_result:
                    st.error(f"❌ Error: {model_result['error']}")
                    continue
                
                # Display sentiment results if available
                if "sentiment" in model_result:
                    sentiment = model_result["sentiment"].get("prediction", "unknown")
                    confidence = model_result["sentiment"].get("confidence", 0.0)
                    
                    # Use colored display based on sentiment
                    if sentiment == "positive":
                        st.markdown(f"**Sentiment:** 🟢 **{sentiment.title()}**")
                    elif sentiment == "negative":
                        st.markdown(f"**Sentiment:** 🔴 **{sentiment.title()}**")
                    else:
                        st.markdown(f"**Sentiment:** ⚪ **{sentiment.title()}**")
                    
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Bar chart for sentiment scores
                    if all(k in model_result["sentiment"] for k in ["positive", "negative", "neutral"]):
                        sentiment_scores = {
                            "Positive": model_result["sentiment"]["positive"],
                            "Neutral": model_result["sentiment"]["neutral"],
                            "Negative": model_result["sentiment"]["negative"]
                        }
                        self.formatter.create_bar_chart(
                            sentiment_scores, f"Sentiment Scores"
                        )
                
                # Display emotion results if available
                if "emotion" in model_result:
                    st.markdown("**Primary Emotion:**")
                    primary_emotion = model_result["emotion"].get("primary_emotion", "unknown")
                    st.markdown(f"**{primary_emotion.title()}**")
                    
                    # Display top emotions if available
                    if "scores" in model_result["emotion"]:
                        emotion_scores = model_result["emotion"]["scores"]
                        # Sort emotions by score
                        sorted_emotions = sorted(
                            emotion_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        # Show top 3 emotions
                        for emotion, score in sorted_emotions[:3]:
                            st.markdown(f"- {emotion.title()}: {score:.2%}")
                
                # Display processing time if available
                if "processing_time" in model_result:
                    st.caption(f"Processing time: {model_result['processing_time']:.3f}s")
    
    def _display_comparison_metrics(self, results: Dict[str, Any]):
        """
        Display metrics comparing the models.
        
        Args:
            results (Dict): Comparison results
        """
        st.markdown("### Comparison Metrics")
        
        # Get comparison metrics
        metrics = results.get("comparison_metrics", {})
        
        if not metrics or all(v is None for v in metrics.values()):
            st.info("No comparison metrics available.")
            return
        
        # Create metrics display
        col1, col2 = st.columns(2)
        
        # Agreement score
        with col1:
            agreement_score = metrics.get("agreement_score")
            if agreement_score is not None:
                # Create a gauge chart for agreement
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=agreement_score * 100,
                    title={"text": "Agreement Score"},
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "blue"},
                        "steps": [
                            {"range": [0, 30], "color": "#EF553B"},
                            {"range": [30, 70], "color": "#FFA15A"},
                            {"range": [70, 100], "color": "#00CC96"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 2},
                            "thickness": 0.75,
                            "value": agreement_score * 100
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                if agreement_score >= 0.7:
                    st.success("✅ High agreement between models")
                elif agreement_score >= 0.4:
                    st.warning("⚠️ Moderate agreement between models")
                else:
                    st.error("❌ Low agreement between models")
        
        # Confidence variance
        with col2:
            confidence_variance = metrics.get("confidence_variance")
            if confidence_variance is not None:
                # Create a gauge chart for confidence variance
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence_variance * 100,
                    title={"text": "Confidence Variance (%)"},
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 25]},
                        "bar": {"color": "blue"},
                        "steps": [
                            {"range": [0, 5], "color": "#00CC96"},
                            {"range": [5, 15], "color": "#FFA15A"},
                            {"range": [15, 25], "color": "#EF553B"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 2},
                            "thickness": 0.75,
                            "value": confidence_variance * 100
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                if confidence_variance < 0.05:
                    st.success("✅ Models have similar confidence levels")
                elif confidence_variance < 0.15:
                    st.warning("⚠️ Moderate variance in confidence levels")
                else:
                    st.error("❌ High variance in confidence levels")
        
        # Key differences
        key_differences = metrics.get("key_differences", [])
        if key_differences:
            st.markdown("### Key Differences")
            for diff in key_differences:
                diff_type = diff.get("type", "unknown")
                description = diff.get("description", "")
                details = diff.get("details", {})
                
                # Display the difference
                st.markdown(f"**{diff_type.capitalize()}:** {description}")
                
                # Create a table for details
                detail_df = pd.DataFrame({
                    "Model": list(details.keys()),
                    "Value": list(details.values())
                })
                st.table(detail_df)
        
        # Recommendation
        recommendation = metrics.get("recommendation")
        if recommendation:
            st.markdown("### Recommendation")
            st.info(f"💡 {recommendation}")
    
    def _display_processing_stats(self, results: Dict[str, Any], model_ids: List[str]):
        """
        Display processing statistics.
        
        Args:
            results (Dict): Comparison results
            model_ids (List[str]): List of model IDs
        """
        st.markdown("### Processing Statistics")
        
        # Collect processing times
        processing_times = {}
        for model_id in model_ids:
            model_result = results.get(model_id, {})
            if "processing_time" in model_result:
                processing_times[model_id] = model_result["processing_time"]
        
        if not processing_times:
            st.info("No processing statistics available.")
            return
        
        # Create a bar chart for processing times
        fig = px.bar(
            x=list(processing_times.keys()),
            y=list(processing_times.values()),
            labels={"x": "Model", "y": "Processing Time (seconds)"},
            title="Processing Time Comparison",
            color=list(processing_times.keys())
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Processing Time (seconds)",
            height=400
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display a table of processing times
        processing_df = pd.DataFrame({
            "Model": [self.comparator.model_names.get(model_id, model_id) for model_id in processing_times.keys()],
            "Processing Time (seconds)": [f"{time:.3f}" for time in processing_times.values()]
        })
        st.table(processing_df)
    
    def _display_agreement_analysis(self, results: Dict[str, Any]):
        """
        Display agreement analysis for batch comparison.
        
        Args:
            results (Dict): Batch comparison results
        """
        st.markdown("### Agreement Analysis")
        
        # Get agreement stats
        agreement_stats = results.get("agreement_stats", {})
        
        if not agreement_stats:
            st.info("No agreement statistics available.")
            return
        
        # Create metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_agreement = agreement_stats.get("sentiment_agreement", 0)
            st.metric("Sentiment Agreement", f"{sentiment_agreement:.1%}")
        
        with col2:
            emotion_agreement = agreement_stats.get("emotion_agreement", 0)
            st.metric("Emotion Agreement", f"{emotion_agreement:.1%}")
        
        with col3:
            full_agreement = agreement_stats.get("full_agreement", 0)
            st.metric("Full Agreement", f"{full_agreement:.1%}")
        
        # Create agreement pie chart
        agreement_data = {
            "Full Agreement": agreement_stats.get("full_agreement", 0),
            "Disagreement": agreement_stats.get("disagreement", 0)
        }
        
        fig = px.pie(
            values=list(agreement_data.values()),
            names=list(agreement_data.keys()),
            title="Agreement Distribution",
            color_discrete_map={
                "Full Agreement": "#00CC96",
                "Disagreement": "#EF553B"
            }
        )
        fig.update_traces(textinfo="percent+label")
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display pairwise agreement
        pairwise_agreement = agreement_stats.get("pairwise_agreement", {})
        
        if pairwise_agreement:
            st.markdown("### Pairwise Agreement")
            
            # Create a DataFrame for pairwise agreement
            pair_data = []
            for pair_key, pair_stats in pairwise_agreement.items():
                # Skip pairs with no comparisons
                if pair_stats.get("total_comparisons", 0) == 0:
                    continue
                
                # Parse model names from pair key
                model1, model2 = pair_key.split("_vs_")
                
                pair_data.append({
                    "Model 1": self.comparator.model_names.get(model1, model1),
                    "Model 2": self.comparator.model_names.get(model2, model2),
                    "Sentiment Agreement": pair_stats.get("sentiment_agreement", 0),
                    "Emotion Agreement": pair_stats.get("emotion_agreement", 0),
                    "Full Agreement": pair_stats.get("both_agreement", 0),
                    "Comparisons": pair_stats.get("total_comparisons", 0)
                })
            
            if pair_data:
                # Create a DataFrame
                pair_df = pd.DataFrame(pair_data)
                
                # Format percentages
                for col in ["Sentiment Agreement", "Emotion Agreement", "Full Agreement"]:
                    pair_df[col] = pair_df[col].apply(lambda x: f"{x:.1%}")
                
                # Display the table
                st.table(pair_df)
                
                # Create a heatmap for full agreement
                model_names = results.get("model_names", [])
                num_models = len(model_names)
                
                if num_models > 1:
                    # Create agreement matrix
                    agreement_matrix = np.zeros((num_models, num_models))
                    
                    # Fill the matrix with agreement scores
                    for i in range(num_models):
                        for j in range(i+1, num_models):
                            model1 = results.get("models_compared", [])[i]
                            model2 = results.get("models_compared", [])[j]
                            pair_key = f"{model1}_vs_{model2}"
                            
                            if pair_key in pairwise_agreement:
                                agreement_score = pairwise_agreement[pair_key].get("both_agreement", 0)
                                agreement_matrix[i, j] = agreement_score
                                agreement_matrix[j, i] = agreement_score
                    
                    # Set diagonal to 1 (self-agreement)
                    np.fill_diagonal(agreement_matrix, 1)
                    
                    # Create heatmap
                    fig = px.imshow(
                        agreement_matrix,
                        labels=dict(x="Model", y="Model", color="Agreement"),
                        x=model_names,
                        y=model_names,
                        color_continuous_scale=["#EF553B", "#FFA15A", "#00CC96"],
                        zmin=0,
                        zmax=1,
                        title="Agreement Heatmap"
                    )
                    
                    # Add text annotations
                    for i in range(num_models):
                        for j in range(num_models):
                            fig.add_annotation(
                                x=i,
                                y=j,
                                text=f"{agreement_matrix[j, i]:.0%}",
                                showarrow=False,
                                font=dict(
                                    color="white" if agreement_matrix[j, i] < 0.7 else "black"
                                )
                            )
                    
                    # Update layout
                    fig.update_layout(height=500)
                    
                    # Display the heatmap
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_model_performance(self, results: Dict[str, Any]):
        """
        Display model performance for batch comparison.
        
        Args:
            results (Dict): Batch comparison results
        """
        st.markdown("### Model Performance")
        
        # Get model performance data
        model_performance = results.get("model_performance", {})
        
        if not model_performance:
            st.info("No model performance data available.")
            return
        
        # Create tabs for different metrics
        perf_tab1, perf_tab2, perf_tab3 = st.tabs([
            "Sentiment Distribution", 
            "Emotion Distribution", 
            "Processing Time"
        ])
        
        with perf_tab1:
            self._display_sentiment_distribution(model_performance, results)
        
        with perf_tab2:
            self._display_emotion_distribution(model_performance, results)
        
        with perf_tab3:
            self._display_batch_processing_stats(model_performance, results)
    
    def _display_sentiment_distribution(self, model_performance: Dict[str, Any], results: Dict[str, Any]):
        """
        Display sentiment distribution for each model.
        
        Args:
            model_performance (Dict): Model performance data
            results (Dict): Full batch comparison results
        """
        st.markdown("### Sentiment Distribution by Model")
        
        # Get model names
        model_names = {model_id: self.comparator.model_names.get(model_id, model_id) 
                      for model_id in model_performance.keys()}
        
        # Create a DataFrame for sentiment distribution
        sentiment_data = []
        
        for model_id, performance in model_performance.items():
            sentiment_dist = performance.get("sentiment_distribution", {})
            
            if sentiment_dist:
                for sentiment, count in sentiment_dist.items():
                    sentiment_data.append({
                        "Model": model_names[model_id],
                        "Sentiment": sentiment.capitalize(),
                        "Count": count
                    })
        
        if sentiment_data:
            # Create DataFrame
            sentiment_df = pd.DataFrame(sentiment_data)
            
            # Create grouped bar chart
            fig = px.bar(
                sentiment_df,
                x="Model",
                y="Count",
                color="Sentiment",
                barmode="group",
                title="Sentiment Distribution by Model",
                color_discrete_map={
                    "Positive": "#00CC96",
                    "Negative": "#EF553B",
                    "Neutral": "#636EFA",
                    "Unknown": "#CCCCCC"
                }
            )
            
            # Update layout
            fig.update_layout(height=400)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Group by model and calculate percentages
            summary_df = sentiment_df.pivot_table(
                index="Model",
                columns="Sentiment",
                values="Count",
                aggfunc="sum"
            ).fillna(0)
            
            # Calculate row totals
            summary_df["Total"] = summary_df.sum(axis=1)
            
            # Calculate percentages
            for col in summary_df.columns:
                if col != "Total":
                    summary_df[f"{col} %"] = summary_df[col] / summary_df["Total"]
            
            # Reorder columns
            ordered_cols = []
            for sentiment in ["Positive", "Neutral", "Negative", "Unknown"]:
                if sentiment in summary_df.columns:
                    ordered_cols.append(sentiment)
                    ordered_cols.append(f"{sentiment} %")
            ordered_cols.append("Total")
            
            final_df = summary_df[ordered_cols].reset_index()
            
            # Format percentages
            for col in final_df.columns:
                if "%" in col:
                    final_df[col] = final_df[col].apply(lambda x: f"{x:.1%}")
            
            # Display the table
            st.table(final_df)
    
    def _display_emotion_distribution(self, model_performance: Dict[str, Any], results: Dict[str, Any]):
        """
        Display emotion distribution for each model.
        
        Args:
            model_performance (Dict): Model performance data
            results (Dict): Full batch comparison results
        """
        st.markdown("### Emotion Distribution by Model")
        
        # Check if emotion analysis was performed
        if results.get("analysis_type") not in ["emotion", "both"]:
            st.info("Emotion analysis was not performed in this batch.")
            return
        
        # Get model names
        model_names = {model_id: self.comparator.model_names.get(model_id, model_id) 
                      for model_id in model_performance.keys()}
        
        # Create a DataFrame for emotion distribution
        emotion_data = []
        
        for model_id, performance in model_performance.items():
            emotion_dist = performance.get("emotion_distribution", {})
            
            if emotion_dist:
                for emotion, count in emotion_dist.items():
                    emotion_data.append({
                        "Model": model_names[model_id],
                        "Emotion": emotion.capitalize(),
                        "Count": count
                    })
        
        if emotion_data:
            # Create DataFrame
            emotion_df = pd.DataFrame(emotion_data)
            
            # Find top emotions across all models
            top_emotions = emotion_df.groupby("Emotion")["Count"].sum().nlargest(5).index.tolist()
            
            # Filter for top emotions
            top_emotion_df = emotion_df[emotion_df["Emotion"].isin(top_emotions)]
            
            # Create grouped bar chart
            fig = px.bar(
                top_emotion_df,
                x="Model",
                y="Count",
                color="Emotion",
                barmode="group",
                title="Top Emotion Distribution by Model"
            )
            
            # Update layout
            fig.update_layout(height=400)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Group by model and calculate percentages
            summary_df = emotion_df.pivot_table(
                index="Model",
                columns="Emotion",
                values="Count",
                aggfunc="sum"
            ).fillna(0)
            
            # Calculate row totals
            summary_df["Total"] = summary_df.sum(axis=1)
            
            # Keep only top emotions
            keep_cols = top_emotions + ["Total"]
            display_df = summary_df[keep_cols].copy()
            
            # Calculate percentages for top emotions
            for emotion in top_emotions:
                display_df[f"{emotion} %"] = display_df[emotion] / display_df["Total"]
            
            # Reorder columns
            ordered_cols = []
            for emotion in top_emotions:
                ordered_cols.append(emotion)
                ordered_cols.append(f"{emotion} %")
            ordered_cols.append("Total")
            
            final_df = display_df[ordered_cols].reset_index()
            
            # Format percentages
            for col in final_df.columns:
                if "%" in col:
                    final_df[col] = final_df[col].apply(lambda x: f"{x:.1%}")
            
            # Display the table
            st.table(final_df)
    
    def _display_batch_processing_stats(self, model_performance: Dict[str, Any], results: Dict[str, Any]):
        """
        Display processing statistics for batch comparison.
        
        Args:
            model_performance (Dict): Model performance data
            results (Dict): Full batch comparison results
        """
        st.markdown("### Processing Performance")
        
        # Get processing stats
        processing_stats = results.get("processing_stats", {})
        
        if not processing_stats:
            st.info("No processing statistics available.")
            return
        
        # Get total processing time and texts processed
        total_time = processing_stats.get("total_time", 0)
        texts_processed = processing_stats.get("texts_processed", 0)
        
        # Display overall metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Processing Time", f"{total_time:.2f}s")
        
        with col2:
            st.metric("Texts Processed", texts_processed)
        
        with col3:
            if total_time > 0:
                processing_rate = texts_processed / total_time
                st.metric("Processing Rate", f"{processing_rate:.2f} texts/s")
        
        # Get model-specific processing times
        model_avg_times = processing_stats.get("model_avg_time", {})
        model_total_times = processing_stats.get("model_total_time", {})
        
        if model_avg_times and model_total_times:
            # Create a DataFrame for processing times
            time_data = []
            
            for model_id in model_avg_times:
                model_name = self.comparator.model_names.get(model_id, model_id)
                avg_time = model_avg_times.get(model_id, 0)
                total_time = model_total_times.get(model_id, 0)
                
                time_data.append({
                    "Model": model_name,
                    "Average Time (s/text)": avg_time,
                    "Total Time (s)": total_time,
                    "Processing Rate (texts/s)": 1 / avg_time if avg_time > 0 else 0
                })
            
            # Create DataFrame and sort by average time
            time_df = pd.DataFrame(time_data).sort_values("Average Time (s/text)")
            
            # Create bar chart for average processing time
            fig = px.bar(
                time_df,
                x="Model",
                y="Average Time (s/text)",
                color="Model",
                title="Average Processing Time per Text",
                labels={"Average Time (s/text)": "Seconds per Text"}
            )
            
            # Update layout
            fig.update_layout(height=400)
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Format processing rate
            time_df["Processing Rate (texts/s)"] = time_df["Processing Rate (texts/s)"].apply(
                lambda x: f"{x:.2f}" if x > 0 else "0.00"
            )
            
            # Display the table
            st.table(time_df)
    
    def _display_detailed_batch_results(self, results: Dict[str, Any]):
        """
        Display detailed results for batch comparison.
        
        Args:
            results (Dict): Batch comparison results
        """
        st.markdown("### Detailed Results")
        
        # Get detailed results
        detailed_results = results.get("detailed_results", [])
        
        if not detailed_results:
            st.info("No detailed results available.")
            return
        
        # Convert to DataFrame
        rows = []
        
        for result in detailed_results:
            # Get text
            text = result.get("text", "")
            
            # Get model results
            model_results = result.get("model_results", {})
            
            # Create row for this text
            row = {"Text": text}
            
            # Add sentiment and emotion for each model
            for model_id, model_result in model_results.items():
                model_name = self.comparator.model_names.get(model_id, model_id)
                
                if "sentiment" in model_result:
                    row[f"{model_name} Sentiment"] = model_result["sentiment"]
                
                if "emotion" in model_result:
                    row[f"{model_name} Emotion"] = model_result["emotion"]
                
                if "error" in model_result:
                    row[f"{model_name} Error"] = model_result["error"]
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Add filter options
        st.markdown("#### Filter Results")
        
        # Get model names
        model_names = results.get("model_names", [])
        
        # Create filter columns
        filter_cols = st.columns(len(model_names))
        active_filters = {}
        
        for i, model_name in enumerate(model_names):
            with filter_cols[i]:
                # Get all sentiment values for this model
                sentiment_col = f"{model_name} Sentiment"
                if sentiment_col in df.columns:
                    sentiment_values = sorted(df[sentiment_col].dropna().unique())
                    
                    # Create multiselect
                    selected = st.multiselect(
                        f"{model_name} Sentiment",
                        options=sentiment_values,
                        default=sentiment_values
                    )
                    
                    if selected:
                        active_filters[sentiment_col] = selected
        
        # Apply filters
        filtered_df = df
        for col, values in active_filters.items():
            filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        # Show results count
        st.write(f"Showing {len(filtered_df)} of {len(df)} results")
        
        # Display the table
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "Download Results as CSV",
            csv,
            file_name="batch_comparison_results.csv",
            mime="text/csv"
        ) 