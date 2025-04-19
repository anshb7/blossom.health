# Blossom Chatbot MVP Implementation
# A hormonal wellness companion using Streamlit and local LLM

import streamlit as st
import json
import datetime
import os
from datetime import datetime, timedelta
import random
import pandas as pd
import plotly.express as px
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# --- Setup ---
# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        "name": "",
        "age": None,
        "cycle_length": None,
        "last_period_date": None,
        "symptoms_log": [],
        "onboarding_complete": False
    }
if 'cycle_day' not in st.session_state:
    st.session_state.cycle_day = None

# --- Functions ---
def calculate_cycle_day():
    """Calculate the current day in menstrual cycle based on last period date"""
    if st.session_state.user_data["last_period_date"]:
        days_since = (datetime.now().date() - st.session_state.user_data["last_period_date"]).days
        cycle_length = st.session_state.user_data["cycle_length"] or 28
        return (days_since % cycle_length) + 1
    return None

def get_phase_info(cycle_day, cycle_length=28):
    """Return information about the current cycle phase based on cycle day"""
    if cycle_day is None:
        return "Unknown"
    
    follicular_phase = int(cycle_length * 0.5)
    ovulatory_phase = int(follicular_phase + 3)
    luteal_phase = cycle_length
    
    if 1 <= cycle_day <= 5:
        return "Menstrual Phase"
    elif 6 <= cycle_day <= follicular_phase:
        return "Follicular Phase"
    elif follicular_phase < cycle_day <= ovulatory_phase:
        return "Ovulatory Phase"
    else:
        return "Luteal Phase"

def log_symptom(symptom, intensity, notes=""):
    """Log a symptom with date and intensity"""
    st.session_state.user_data["symptoms_log"].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "cycle_day": st.session_state.cycle_day,
        "phase": get_phase_info(st.session_state.cycle_day),
        "symptom": symptom,
        "intensity": intensity,
        "notes": notes
    })
    save_user_data()

def identify_patterns():
    """Analyze symptoms log to identify patterns"""
    if len(st.session_state.user_data["symptoms_log"]) < 3:
        return "Not enough data to identify patterns yet."
    
    # Simple pattern analysis for demonstration
    symptoms_df = pd.DataFrame(st.session_state.user_data["symptoms_log"])
    
    # Example pattern: Check if certain symptoms occur more in certain phases
    if "phase" in symptoms_df.columns and len(symptoms_df) > 0:
        phase_symptoms = symptoms_df.groupby(["phase", "symptom"])["intensity"].mean().reset_index()
        high_intensity = phase_symptoms[phase_symptoms["intensity"] > 3]
        
        if len(high_intensity) > 0:
            pattern = f"You tend to experience higher intensity {high_intensity.iloc[0]['symptom']} " \
                     f"during your {high_intensity.iloc[0]['phase']}."
            return pattern
    
    return "Still analyzing your patterns. Keep logging your symptoms!"

def save_user_data():
    """Save user data to session state"""
    # In a real implementation, you would save to a database here
    pass

def generate_insight():
    """Generate personalized insight based on user data and patterns"""
    # Simple insights for demonstration
    insights = [
        "Your mood symptoms seem to intensify during the luteal phase. Magnesium-rich foods like dark chocolate might help.",
        "I notice your energy levels drop during menstruation. Gentle movement like walking or stretching could help maintain energy.",
        "Your bloating symptoms appear consistently before your period. Reducing salt intake and increasing water might help.",
        "You've reported headaches in your follicular phase. Ensuring adequate hydration may help reduce their frequency.",
        "Your sleep quality decreases before menstruation. A consistent bedtime routine might be beneficial."
    ]
    
    # In a real implementation, you would use the LLM to generate truly personalized insights
    return random.choice(insights)

# --- LLM Setup ---
def setup_llm():
    """Setup the LLM for chat interactions"""
    try:
        llm = Ollama(base_url="https://a0d6-2a09-bac1-36c0-40-00-243-6.ngrok-free.app",model="llama3")
        
        template = """
        You are Bloom, an empathetic AI assistant specializing in women's hormonal health and wellness.
        You're speaking with a woman who may be experiencing hormonal issues like PCOS, mood swings, 
        irregular periods, or other symptoms. Be gentle, supportive, and knowledgeable.

        Current conversation:
        {chat_history}

        User information:
        - Current cycle day: {cycle_day}
        - Current phase: {cycle_phase}
        - Recent symptoms: {recent_symptoms}

        User: {human_input}
        Blossom:
        """
        
        prompt = PromptTemplate(
            input_variables=["chat_history", "cycle_day", "cycle_phase", "recent_symptoms", "human_input"],
            template=template
        )
        
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
        
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )
        
        return chain
    except Exception as e:
        print(f"Error setting up LLM: {e}")
        return None

# --- UI Components ---
def render_onboarding():
    """Render the onboarding flow"""
    st.title("Welcome to Blossom 🌸")
    st.write("Let's get to know you so I can provide personalized support.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("What's your name?", value=st.session_state.user_data["name"])
        age = st.number_input("Age", min_value=18, max_value=65, value=st.session_state.user_data["age"] or 30)
    
    with col2:
        cycle_length = st.number_input("Average cycle length (days)", min_value=21, max_value=40, 
                                     value=st.session_state.user_data["cycle_length"] or 28)
        last_period = st.date_input("First day of your last period", 
                                  value=st.session_state.user_data["last_period_date"] or datetime.now().date())
    
    if st.button("Complete Profile"):
        st.session_state.user_data.update({
            "name": name,
            "age": age,
            "cycle_length": cycle_length,
            "last_period_date": last_period,
            "onboarding_complete": True
        })
        st.session_state.cycle_day = calculate_cycle_day()
        
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Hi {name}! I'm Bloom, your hormonal wellness companion. How are you feeling today?"
        })
        st.rerun()

def render_symptom_tracker():
    """Render the symptom tracking sidebar"""
    st.sidebar.title("Quick Symptom Log")
    
    symptom = st.sidebar.selectbox(
        "What are you experiencing?",
        ["Mood swings", "Anxiety", "Fatigue", "Bloating", "Cramps", "Headache", "Acne", "Cravings", "Other"]
    )
    
    intensity = st.sidebar.slider("Intensity", 1, 5, 3)
    
    notes = st.sidebar.text_area("Notes (optional)")
    
    if st.sidebar.button("Log Symptom"):
        log_symptom(symptom, intensity, notes)
        st.sidebar.success("Symptom logged!")
        
        # Sometimes offer an insight
        if len(st.session_state.user_data["symptoms_log"]) % 3 == 0:
            insight = generate_insight()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"**Blossom Insight**: {insight}"
            })
            st.rerun()
def render_symptom_dashboard():
    st.subheader("📊 Symptom Trends")

    symptoms_df = pd.DataFrame(st.session_state.user_data["symptoms_log"])

    if symptoms_df.empty:
        st.info("Start logging symptoms to view trends.")
        return

    # Convert date column to datetime for sorting
    symptoms_df['date'] = pd.to_datetime(symptoms_df['date'])

    # Line chart: symptom intensity over time
    fig1 = px.line(
        symptoms_df,
        x="date",
        y="intensity",
        color="symptom",
        title="Symptom Intensity Over Time",
        markers=True
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Bar chart: average intensity by phase
    fig2 = px.bar(
        symptoms_df.groupby(["phase", "symptom"])["intensity"].mean().reset_index(),
        x="phase",
        y="intensity",
        color="symptom",
        barmode="group",
        title="Average Symptom Intensity by Cycle Phase"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: symptom frequency per cycle day
    heat_df = symptoms_df.groupby(["cycle_day", "symptom"]).size().reset_index(name='count')
    fig3 = px.density_heatmap(
        heat_df,
        x="cycle_day",
        y="symptom",
        z="count",
        color_continuous_scale="Purples",
        title="Symptom Frequency Heatmap by Cycle Day"
    )
    st.plotly_chart(fig3, use_container_width=True)

def render_dashboard():
    """Render the user dashboard with cycle and symptom information"""
    st.sidebar.title("Your Cycle")
    
    cycle_day = st.session_state.cycle_day
    if cycle_day:
        st.sidebar.metric("Current Cycle Day", cycle_day)
        phase = get_phase_info(cycle_day)
        st.sidebar.write(f"You're in your **{phase}**")
    
    st.sidebar.divider()
    render_symptom_tracker()
    
    # Display recent logs
    if len(st.session_state.user_data["symptoms_log"]) > 0:
        st.sidebar.divider()
        st.sidebar.subheader("Recent Logs")
        logs = st.session_state.user_data["symptoms_log"][-3:]  # Show last 3 logs
        for log in reversed(logs):
            st.sidebar.caption(f"{log['date']} - Day {log['cycle_day']}")
            st.sidebar.write(f"**{log['symptom']}** (Intensity: {log['intensity']})")
            if log['notes']:
                st.sidebar.write(f"*{log['notes']}*")
        # Export logs
    if len(st.session_state.user_data["symptoms_log"]) > 0:
        st.sidebar.divider()
        st.sidebar.subheader("Download Your Logs")

        logs_df = pd.DataFrame(st.session_state.user_data["symptoms_log"])

        csv = logs_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name='blossom_symptom_log.csv',
            mime='text/csv'
        )

        json_data = json.dumps(st.session_state.user_data["symptoms_log"], indent=2)
        st.sidebar.download_button(
            label="🧾 Download as JSON",
            data=json_data,
            file_name='blossom_symptom_log.json',
            mime='application/json'
        )


def render_chat_interface():
    """Render the chat interface"""
    st.title("Blossom 🌸")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("How are you feeling today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get recent symptoms for context
                recent_symptoms = "No recent symptoms logged."
                if st.session_state.user_data["symptoms_log"]:
                    recent = st.session_state.user_data["symptoms_log"][-3:]
                    symptom_list = [f"{r['symptom']} (intensity: {r['intensity']})" for r in recent]
                    recent_symptoms = ", ".join(symptom_list)
                
                # Try using LLM if available
                llm_chain = setup_llm()
                if llm_chain:
                    response = llm_chain.predict(
                        human_input=prompt,
                        cycle_day=st.session_state.cycle_day or "Unknown",
                        cycle_phase=get_phase_info(st.session_state.cycle_day) or "Unknown",
                        recent_symptoms=recent_symptoms
                    )
                else:
                    # Fallback responses if LLM is not available
                    responses = [
                        f"I understand how challenging hormonal symptoms can be. Based on your cycle day ({st.session_state.cycle_day}), this is normal. How can I support you today?",
                        "Thank you for sharing how you're feeling. Would you like some suggestions for managing these symptoms?",
                        "I'm here to listen and help. Have you noticed any patterns with these symptoms?",
                        "That sounds difficult. Many women experience similar challenges. Would gentle movement or stress reduction techniques help right now?"
                    ]
                    response = random.choice(responses)
                    
                    # Check for specific keywords
                    if "tired" in prompt or "exhausted" in prompt or "fatigue" in prompt:
                        response = "Fatigue can be common, especially during your luteal phase. Would you like some energy-boosting tips that align with your current cycle phase?"
                    elif "anxious" in prompt or "anxiety" in prompt or "worried" in prompt:
                        response = "I notice you're feeling anxious. This is common during hormonal fluctuations. Deep breathing or a short walk might help regulate your nervous system."
                    elif "cramps" in prompt or "pain" in prompt:
                        response = "I'm sorry you're experiencing pain. A warm compress and anti-inflammatory foods might provide some relief. Would you like more specific suggestions?"
                    
                st.markdown(response)
                
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            # Render Symptom Visualization if logs exist
            # Only show dashboard after 5 symptom logs
    if len(st.session_state.user_data["symptoms_log"]) >= 5:
            render_symptom_dashboard()
    else:
            st.markdown(
                "<div style='margin-top: 2rem; font-size: 1.1rem; color: #888;'>"
                "📊 Log a few more symptoms to unlock your personal dashboard."
                "</div>", unsafe_allow_html=True
            )



# --- Main App ---
def main():
    st.set_page_config(
        page_title="Blossom - Hormonal Wellness Companion",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS for lavender theme
    st.markdown("""
    <style>
        :root {
            --primary-color: #9d8cd4;
        }
        .stButton>button {
            background-color: #9d8cd4;
            color: white;
        }
        .stTextInput>div>div>input {
            border-color: #9d8cd4;
        }
        .stHeader {
            color: #9d8cd4;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if user needs onboarding
    if not st.session_state.user_data["onboarding_complete"]:
        render_onboarding()
    else:
        # Update cycle day
        st.session_state.cycle_day = calculate_cycle_day()
        
        # Render main interface
        render_dashboard()
        render_chat_interface()

if __name__ == "__main__":
    main()