import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

try:
    from src.logging.base import WorkflowLog
    from src.logging.json_logger import JsonLogger
except ImportError:
    # Fallback for different execution contexts
    from logging.base import WorkflowLog
    from logging.json_logger import JsonLogger

class LogVisualizer:
    def __init__(self, log_dir: str = "logs"):
        self.logger = JsonLogger(log_dir)
    
    def get_workflow_summary(self, days: int = 7) -> pd.DataFrame:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        workflows = self.logger.get_workflow_logs(start_time=start_time)
        
        data = []
        for wf in workflows:
            data.append({
                'workflow_id': wf.workflow_id,  # Access as object attributes
                'query': wf.query,
                'success': wf.success,
                'timestamp': wf.start_time,
                'num_steps': len(wf.step_ids)
            })
        
        return pd.DataFrame(data)
    
    def get_step_performance(self, days: int = 7) -> pd.DataFrame:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        workflows = self.logger.get_workflow_logs(start_time=start_time)
        
        data = []
        for wf in workflows:
            for step_id in wf.step_ids:
                with open(self.logger.step_dir / f"{step_id}.json") as f:
                    step = json.load(f)
                    data.append({
                        'workflow_id': wf.workflow_id,
                        'step_name': step['step_name'],
                        'success': step['success'],
                        'duration_ms': step['duration_ms'],
                        'timestamp': datetime.fromisoformat(step['timestamp'])
                    })
        
        return pd.DataFrame(data)

def main():
    st.set_page_config(page_title="RAG Pipeline Monitor", layout="wide")
    st.title("RAG Pipeline Monitor")
    
    # Initialize visualizer
    viz = LogVisualizer()
    
    # Time range selector
    days = st.slider("Select time range (days)", 1, 30, 7)
    
    # Get data
    workflow_df = viz.get_workflow_summary(days)
    
    if len(workflow_df) == 0:
        st.warning("No workflow logs found in the selected time range")
        return
        
    step_df = viz.get_step_performance(days)
    
    # Dashboard layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Workflow Success Rate")
        success_rate = workflow_df['success'].mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
        
        fig = px.pie(workflow_df, names='success', title='Workflow Success Distribution')
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Step Performance")
        if len(step_df) > 0:
            fig = px.box(step_df, x='step_name', y='duration_ms', 
                        title='Step Duration Distribution')
            st.plotly_chart(fig)
        else:
            st.info("No step data available")
    
    # Timeline view
    st.subheader("Workflow Timeline")
    fig = px.scatter(workflow_df, x='timestamp', y='num_steps',
                    color='success', hover_data=['query'],
                    title='Workflow Executions Over Time')
    st.plotly_chart(fig)
    
    # Step success rates
    if len(step_df) > 0:
        st.subheader("Step Success Rates")
        step_success = step_df.groupby('step_name')['success'].mean() * 100
        fig = px.bar(step_success, title='Step Success Rates (%)')
        st.plotly_chart(fig)
    
    # Recent workflows table
    st.subheader("Recent Workflows")
    st.dataframe(
        workflow_df.sort_values('timestamp', ascending=False)
        .head(10)[['workflow_id', 'query', 'success', 'timestamp']]
    )

if __name__ == "__main__":
    main() 