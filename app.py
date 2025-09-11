import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="EduGuardians - Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .risk-high { background-color: #fee2e2; padding: 10px; border-radius: 5px; }
    .risk-medium { background-color: #fef3c7; padding: 10px; border-radius: 5px; }
    .risk-low { background-color: #d1fae5; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

class EduGuardians:
    def __init__(self):
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'students_data' not in st.session_state:
            st.session_state.students_data = None
    
    def load_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        student_ids = [f"STU{str(i+1).zfill(3)}" for i in range(30)]
        names = ["Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Singh", "Vikram Gupta"] * 6
        
        # Attendance data
        attendance_data = {
            'student_id': student_ids,
            'name': names[:30],
            'class': np.random.choice(['10A', '10B', '11A', '11B'], 30),
            'attendance_pct': np.random.normal(75, 15, 30).clip(40, 100)
        }
        
        # Assessment data
        assessment_records = []
        for sid in student_ids:
            for i in range(np.random.randint(3, 8)):
                assessment_records.append({
                    'student_id': sid,
                    'test_id': f"TEST{str(i+1).zfill(3)}",
                    'test_date': (datetime.now() - timedelta(days=np.random.randint(1, 180))).strftime('%Y-%m-%d'),
                    'score': np.random.normal(70, 20, 1)[0].clip(0, 100),
                    'subject': np.random.choice(['Math', 'Physics', 'Chemistry', 'Biology']),
                    'attempt_no': np.random.choice([1, 1, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.05, 0.05])
                })
        
        # Fee data
        fee_data = {
            'student_id': student_ids,
            'amount_due': np.random.exponential(3000, 30).clip(0, 20000),
            'last_payment_date': [(datetime.now() - timedelta(days=np.random.randint(0, 120))).strftime('%Y-%m-%d') for _ in range(30)]
        }
        
        return pd.DataFrame(attendance_data), pd.DataFrame(assessment_records), pd.DataFrame(fee_data)
    
    def process_data(self, attendance_df, assessment_df, fee_df):
        """Process and merge all data sources"""
        try:
            # Process attendance
            attendance_processed = attendance_df.groupby('student_id').agg({
                'name': 'first',
                'class': 'first',
                'attendance_pct': 'mean'
            }).reset_index()
            
            # Process assessments
            assessment_processed = assessment_df.groupby('student_id').agg({
                'score': ['mean', lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0],
                'attempt_no': 'max'
            }).reset_index()
            assessment_processed.columns = ['student_id', 'avg_score', 'score_trend', 'max_attempts']
            
            # Process fees
            fee_processed = fee_df.groupby('student_id').agg({
                'amount_due': 'sum'
            }).reset_index()
            
            # Merge all data
            merged_df = attendance_processed.merge(assessment_processed, on='student_id', how='outer')
            merged_df = merged_df.merge(fee_processed, on='student_id', how='outer')
            
            # Fill missing values
            merged_df = merged_df.fillna({
                'attendance_pct': 75, 'avg_score': 60, 'score_trend': 0,
                'max_attempts': 1, 'amount_due': 0
            })
            
            return merged_df
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None
    
    def calculate_risk_score(self, df, thresholds):
        """Calculate rule-based risk scores"""
        risk_scores = []
        risk_labels = []
        explanations = []
        
        for _, row in df.iterrows():
            score = 0
            explanation = []
            
            # Attendance factor
            if row['attendance_pct'] < thresholds['attendance_med']:
                score += 30
                explanation.append(f"Low attendance ({row['attendance_pct']:.1f}%)")
            elif row['attendance_pct'] < thresholds['attendance_high']:
                score += 15
            
            # Score factor
            if row['avg_score'] < thresholds['score_med']:
                score += 25
                explanation.append(f"Poor academic performance ({row['avg_score']:.1f}%)")
            elif row['avg_score'] < thresholds['score_high']:
                score += 10
            
            # Attempts factor
            if row['max_attempts'] >= thresholds['attempts_high']:
                score += 20
                explanation.append(f"Multiple test attempts ({row['max_attempts']})")
            
            # Fee factor
            if row['amount_due'] > thresholds['fee_threshold']:
                score += 25
                explanation.append(f"Outstanding fees (₹{row['amount_due']:,.0f})")
            
            # Determine risk level
            if score >= 50:
                risk_label = 'High'
            elif score >= 25:
                risk_label = 'Medium'
            else:
                risk_label = 'Low'
            
            if not explanation:
                explanation.append("No significant risk factors")
            
            risk_scores.append(min(score, 100))
            risk_labels.append(risk_label)
            explanations.append("; ".join(explanation))
        
        df['risk_score'] = risk_scores
        df['risk_label'] = risk_labels
        df['explanation'] = explanations
        
        return df
    
    def train_model(self, df):
        """Train ML model for dropout prediction"""
        try:
            # Prepare features
            features = ['attendance_pct', 'avg_score', 'score_trend', 'max_attempts', 'amount_due']
            X = df[features]
            y = (df['risk_label'] == 'High').astype(int)
            
            if len(np.unique(y)) < 2:
                st.warning("Not enough variation in target variable for ML training")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Handle class imbalance with SMOTE
            if len(np.unique(y_train)) > 1:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            joblib.dump(model, 'model.pkl')
            
            return model, accuracy, classification_report(y_test, y_pred)
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    def run(self):
        # Header
        st.markdown('<div class="main-header"><h1>🎓 EduGuardians</h1><p>AI-Powered Dropout Prediction & Student Counseling System</p></div>', unsafe_allow_html=True)
        
        # Sidebar - Configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Risk thresholds
            st.subheader("Risk Thresholds")
            thresholds = {
                'attendance_high': st.slider("Attendance High", 70, 95, 85),
                'attendance_med': st.slider("Attendance Medium", 50, 80, 70),
                'score_high': st.slider("Score High", 60, 90, 80),
                'score_med': st.slider("Score Medium", 40, 70, 60),
                'attempts_high': st.slider("Max Attempts", 2, 5, 3),
                'fee_threshold': st.slider("Fee Threshold (₹)", 1000, 20000, 5000)
            }
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Upload Data", "🔮 Predict", "🤖 ML Training"])
        
        with tab2:
            st.header("Data Upload")
            
            use_sample = st.checkbox("Use sample data for demonstration")
            
            if use_sample:
                if st.button("Load Sample Data"):
                    attendance_df, assessment_df, fee_df = self.load_sample_data()
                    merged_df = self.process_data(attendance_df, assessment_df, fee_df)
                    if merged_df is not None:
                        st.session_state.students_data = self.calculate_risk_score(merged_df, thresholds)
                        st.success(f"Loaded sample data for {len(merged_df)} students")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Attendance CSV")
                    attendance_file = st.file_uploader("Upload attendance data", type=['csv'], key='attendance')
                
                with col2:
                    st.subheader("Assessments CSV")
                    assessment_file = st.file_uploader("Upload assessment data", type=['csv'], key='assessments')
                
                with col3:
                    st.subheader("Fees CSV")
                    fee_file = st.file_uploader("Upload fee data", type=['csv'], key='fees')
                
                if st.button("Process Data") and attendance_file and assessment_file and fee_file:
                    try:
                        attendance_df = pd.read_csv(attendance_file)
                        assessment_df = pd.read_csv(assessment_file)
                        fee_df = pd.read_csv(fee_file)
                        
                        merged_df = self.process_data(attendance_df, assessment_df, fee_df)
                        if merged_df is not None:
                            st.session_state.students_data = self.calculate_risk_score(merged_df, thresholds)
                            st.success(f"Successfully processed data for {len(merged_df)} students")
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        
        # Show dashboard if data is loaded
        if st.session_state.students_data is not None:
            df = st.session_state.students_data
            
            with tab1:
                st.header("Risk Analytics Dashboard")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Students", len(df))
                with col2:
                    high_risk = len(df[df['risk_label'] == 'High'])
                    st.metric("High Risk", high_risk, delta=None if high_risk == 0 else f"{high_risk/len(df)*100:.1f}%")
                with col3:
                    medium_risk = len(df[df['risk_label'] == 'Medium'])
                    st.metric("Medium Risk", medium_risk)
                with col4:
                    low_risk = len(df[df['risk_label'] == 'Low'])
                    st.metric("Low Risk", low_risk)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk distribution
                    risk_counts = df['risk_label'].value_counts()
                    fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                                title="Risk Level Distribution", color_discrete_map={
                                    'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'
                                })
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Attendance vs Score scatter
                    fig = px.scatter(df, x='attendance_pct', y='avg_score', color='risk_label',
                                   title="Attendance vs Academic Performance",
                                   color_discrete_map={
                                       'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'
                                   })
                    st.plotly_chart(fig, use_container_width=True)
                
                # Student table
                st.subheader("Student Details")
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    risk_filter = st.selectbox("Filter by Risk", ['All', 'High', 'Medium', 'Low'])
                with col2:
                    search_term = st.text_input("Search students", "")
                
                # Apply filters
                filtered_df = df.copy()
                if risk_filter != 'All':
                    filtered_df = filtered_df[filtered_df['risk_label'] == risk_filter]
                if search_term:
                    filtered_df = filtered_df[
                        filtered_df['name'].str.contains(search_term, case=False, na=False) |
                        filtered_df['student_id'].str.contains(search_term, case=False, na=False)
                    ]
                
                # Display table with styling
                def style_risk(val):
                    if val == 'High':
                        return 'background-color: #fee2e2'
                    elif val == 'Medium':
                        return 'background-color: #fef3c7'
                    else:
                        return 'background-color: #d1fae5'
                
                styled_df = filtered_df[['student_id', 'name', 'attendance_pct', 'avg_score', 
                                       'max_attempts', 'amount_due', 'risk_score', 'risk_label']].style.applymap(
                    style_risk, subset=['risk_label'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Export button
                if st.button("📥 Export Risk Report"):
                    csv = df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "risk_report.csv", "text/csv")
            
            with tab3:
                st.header("Quick Risk Prediction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Enter Student Details")
                    
                    student_id = st.text_input("Student ID", "STU999")
                    name = st.text_input("Name", "Test Student")
                    attendance = st.slider("Attendance %", 0.0, 100.0, 75.0)
                    avg_score = st.slider("Average Score %", 0.0, 100.0, 70.0)
                    max_attempts = st.slider("Max Test Attempts", 1, 5, 2)
                    amount_due = st.number_input("Amount Due (₹)", 0, 50000, 2000)
                    
                    if st.button("🔮 Predict Risk"):
                        # Create prediction data
                        pred_data = pd.DataFrame([{
                            'student_id': student_id,
                            'name': name,
                            'attendance_pct': attendance,
                            'avg_score': avg_score,
                            'score_trend': 0,
                            'max_attempts': max_attempts,
                            'amount_due': amount_due
                        }])
                        
                        # Calculate risk
                        pred_result = self.calculate_risk_score(pred_data, thresholds)
                        
                        # Store in session state for display
                        st.session_state.prediction_result = pred_result.iloc[0]
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    if hasattr(st.session_state, 'prediction_result'):
                        result = st.session_state.prediction_result
                        
                        # Risk score and label
                        risk_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[result['risk_label']]
                        st.metric("Risk Level", f"{risk_color} {result['risk_label']}", 
                                f"Score: {result['risk_score']}/100")
                        
                        # ML prediction if model exists
                        if st.session_state.model:
                            features = [result['attendance_pct'], result['avg_score'], 
                                      result['score_trend'], result['max_attempts'], result['amount_due']]
                            ml_prob = st.session_state.model.predict_proba([features])[0][1]
                            st.metric("ML Prediction", f"{ml_prob:.1%}", "Dropout Probability")
                        
                        # Explanation
                        st.subheader("Risk Factors")
                        st.write(result['explanation'])
                        
                        # Recommendations
                        if result['risk_label'] != 'Low':
                            st.subheader("Recommended Actions")
                            recommendations = []
                            if result['attendance_pct'] < thresholds['attendance_med']:
                                recommendations.append("📞 Schedule attendance counseling")
                            if result['avg_score'] < thresholds['score_med']:
                                recommendations.append("📚 Provide additional academic support")
                            if result['amount_due'] > thresholds['fee_threshold']:
                                recommendations.append("💳 Contact for fee payment plan")
                            if result['risk_label'] == 'High':
                                recommendations.append("🚨 Immediate intervention required")
                            
                            for rec in recommendations:
                                st.write(f"• {rec}")
            
            with tab4:
                st.header("ML Model Training")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Train New Model")
                    
                    if st.button("🤖 Train RandomForest Model"):
                        with st.spinner("Training model..."):
                            result = self.train_model(df)
                            
                            if result:
                                model, accuracy, report = result
                                st.session_state.model = model
                                
                                st.success(f"Model trained successfully!")
                                st.metric("Accuracy", f"{accuracy:.2%}")
                                
                                st.subheader("Classification Report")
                                st.text(report)
                
                with col2:
                    st.subheader("Model Status")
                    
                    if os.path.exists('model.pkl'):
                        if st.button("📁 Load Existing Model"):
                            st.session_state.model = joblib.load('model.pkl')
                            st.success("Model loaded successfully!")
                    
                    if st.session_state.model:
                        st.success("✅ Model is ready")
                        
                        # Feature importance
                        if hasattr(st.session_state.model, 'feature_importances_'):
                            features = ['Attendance %', 'Avg Score', 'Score Trend', 'Max Attempts', 'Amount Due']
                            importance = st.session_state.model.feature_importances_
                            
                            fig = px.bar(x=features, y=importance, title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No model loaded. Train a new model or load existing one.")

if __name__ == "__main__":
    app = EduGuardians()
    app.run()
