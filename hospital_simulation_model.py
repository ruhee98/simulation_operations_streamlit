import simpy
import random
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


# Constants
GENERAL_DURATION = st.sidebar.slider("Mean Duration of General Consultation", min_value=30, max_value=60, value=45)
SURGERY_DURATION = st.sidebar.slider("Mean Duration of Surgery Procedure", min_value=0, max_value=120, value=90)
OBSERVATION_DURATION = st.sidebar.slider("Mean Duration of Testing/Observation Procedure", min_value=0, max_value=90, value=60)
NUM_GENERAL_ROOMS = st.sidebar.slider("Number of General Rooms", min_value=1, max_value=10, value=2)
NUM_SURGERY_ROOMS = st.sidebar.slider("Number of Surgery Rooms", min_value=1, max_value=10, value=2)
NUM_OBSERVATION_ROOMS = st.sidebar.slider("Number of Observation Rooms", min_value=1, max_value=10, value=2)
TIME_SLOT_MIN = st.sidebar.slider("Time Slot Intervals (in min)", min_value=0, max_value=60, value=15)
SIM_DURATION_DAYS = st.sidebar.slider("Number of Days", min_value=0, max_value=31, value=7)

start_treatment_datetime = datetime(2024, 7, 1, 0, 0)


# Initialize environment and global variables
env = simpy.Environment()
patient_data = []
summary_data = []
summary_metrics_data = []
arrival_times = []
room_usages = {'General': [], 'Observation': [], 'Surgery': []}
transition_counts = {'General': {'General': 0, 'Observation': 0, 'Surgery': 0},
                     'Observation': {'General': 0, 'Observation': 0, 'Surgery': 0},
                     'Surgery': {'General': 0, 'Observation': 0, 'Surgery': 0}}


patient_id = 0
patients_in_queue = 0
patients_in_general_queue = 0
patients_in_observation_queue = 0
patients_in_surgery_queue = 0
patients_discharged = 0

# Probabilities
prob_observation_after_general = 0.5
prob_stay_general_after_general = 0.2
prob_surgery_after_general = 0.2
prob_discharge_after_general = 0.1
prob_surgery_after_observation = 0.5
prob_discharge_after_observation = 0.5

class Hospital:
    def __init__(self, env, num_general_rooms, num_observation_rooms, num_surgery_rooms):
        self.env = env
        self.general_room = simpy.Resource(env, capacity=num_general_rooms)
        self.observation_room = simpy.Resource(env, capacity=num_observation_rooms)
        self.surgery_room = simpy.Resource(env, capacity=num_surgery_rooms)
        self.queue_counts = {'General': 0, 'Observation': 0, 'Surgery': 0, 'Discharge': 0}


class Patient:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.procedure_type = 'General'
        self.priority = None
        self.wait_time = 0
        self.duration = 0
        self.next_procedure_type = None

    def set_priority(self):
        self.priority = random.choices(['Priority 1', 'Priority 2', 'Priority 3'], weights=[0.3, 0.5, 0.2], k=1)[0]
        return self.priority
    
    def set_patient_outcome(self):
        if self.priority == 'Priority 1':
            self.procedure_type = 'Surgery'
        elif self.priority == 'Priority 2': 
            self.procedure_type = random.choices(['General', 'Observation'], [0.4, 0.6])[0]
        else:
            self.procedure_type = 'Discharge'
        return self.procedure_type

def assign_room(env, arrival_datetime, hospital, patient):
    global patients_in_queue, last_treatment_end_time, patients_discharged, arrival_times, room_usages
    
    # Track patient arrivals by hour and day of the week
    arrival_times.append(arrival_datetime)
    
    delay = random.randint(30, 60)
    yield env.timeout(delay)

    treatment_start_time = arrival_datetime + timedelta(minutes=delay)
    
    patient = Patient(patient_id)
    priority = patient.set_priority()
    procedure_type = patient.set_patient_outcome()
    
    if procedure_type == 'Discharge':
        patients_in_queue -= 1
        patients_discharged += 1
        hospital.queue_counts['Discharge'] += 1
        print(f"Patient {patient.patient_id} discharged. Patients in queue after decrement: {patients_in_queue}")
        print(f"Updated Counts: General Queue: {hospital.queue_counts['General']}, Observation Queue: {hospital.queue_counts['Observation']}, Surgery Queue: {hospital.queue_counts['Surgery']}, Discharge: {hospital.queue_counts['Discharge']}")
        return

    duration = GENERAL_DURATION if procedure_type == 'General' else OBSERVATION_DURATION if procedure_type == 'Observation' else SURGERY_DURATION
    patient.duration = duration

    treatment_end_time = treatment_start_time + timedelta(minutes=duration)
    last_treatment_end_time = max(last_treatment_end_time, treatment_end_time)

    wait_time = (treatment_start_time - arrival_datetime).total_seconds() / 60
    patient.wait_time = wait_time
    
    patients_in_queue += 1
    hospital.queue_counts[procedure_type] += 1
    print(f"General Queue: {hospital.queue_counts['General']}, Observation Queue: {hospital.queue_counts['Observation']}, Surgery Queue: {hospital.queue_counts['Surgery']}")

    room_request = None
    room_start_time = None
    if procedure_type == 'General':
        room_request = hospital.general_room.request()
        room_start_time = treatment_start_time
    elif procedure_type == 'Observation':
        room_request = hospital.observation_room.request()
        room_start_time = treatment_start_time
    elif procedure_type == 'Surgery':
        room_request = hospital.surgery_room.request()
        room_start_time = treatment_start_time

    if room_request:
        with room_request as req:
            yield req
            patients_in_queue -= 1
            hospital.queue_counts[procedure_type] -= 1
            print(f"Patient {patient.patient_id} enters {procedure_type} room. Patients in queue after decrement: {patients_in_queue}")
            print(f"Updated Counts: General Queue: {hospital.queue_counts['General']}, Observation Queue: {hospital.queue_counts['Observation']}, Surgery Queue: {hospital.queue_counts['Surgery']}")

            # Log room usage
            if procedure_type != 'Discharge':
                room_usage_time = room_start_time
                room_usage_type = procedure_type
                room_usages[room_usage_type].append(room_usage_time)
    
    log_patient_data(patient.patient_id, arrival_datetime, priority, procedure_type, treatment_start_time, treatment_end_time, wait_time, duration)
    
    # Determine next procedure type
    if procedure_type == 'General':
        next_procedure = random.choices(
            ['Observation', 'Surgery', 'Discharge'], 
            weights=[prob_observation_after_general, prob_surgery_after_general, prob_discharge_after_general],
            k=1
        )[0]
    elif procedure_type == 'Observation':
        next_procedure = random.choices(
            ['Discharge', 'General', 'Surgery'], 
            weights=[prob_discharge_after_observation, 0.5 * prob_discharge_after_observation, prob_surgery_after_observation],
            k=1
        )[0]
    else:
        next_procedure = 'Discharge'
    
    if next_procedure != 'Discharge':
        transition_counts[procedure_type][next_procedure] += 1
        patient.next_procedure_type = next_procedure
    
    yield env.timeout(duration)
    
    return patients_in_queue, hospital.queue_counts['General'], hospital.queue_counts['Observation'], hospital.queue_counts['Surgery'], arrival_times, room_usages

def log_patient_data(patient_id, arrival_datetime, priority, procedure_type, treatment_start_time, treatment_end_time, wait_time, duration):
    global patient_data
    log_entry = {
        'patient_id': patient_id,
        'arrival_time': arrival_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'priority': priority,
        'procedure_type': procedure_type,
        'treatment_start_time': treatment_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'treatment_end_time': treatment_end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'wait_time': wait_time,
        'duration': duration
    }
    patient_data.append(log_entry)
    

def generate_arrival_times(start_datetime, end_datetime, min_patients_per_hour, max_patients_per_hour, time_slot_min):
    arrival_times = []
    current_time = start_datetime

    while current_time < end_datetime:
        num_patients = random.randint(min_patients_per_hour, max_patients_per_hour)
        for _ in range(num_patients):
            # Generate arrival time within the current time slot
            inter_arrival_minutes = random.randint(0, time_slot_min - 1)  # Random minutes within the time slot
            next_time = current_time + timedelta(minutes=inter_arrival_minutes)
            if next_time >= end_datetime:
                break
            arrival_times.append(next_time)
            current_time = next_time

        # Move to the next time slot
        current_time = current_time + timedelta(minutes=time_slot_min)
        if current_time > end_datetime:
            break

    return sorted(arrival_times)

def run_simulation():
    global patient_id, patients_in_queue, summary_data, last_treatment_end_time, summary_metrics_data, patients_discharged, arrival_times, start_treatment_datetime, room_usages

    env = simpy.Environment()
    hospital = Hospital(env, NUM_GENERAL_ROOMS, NUM_OBSERVATION_ROOMS, NUM_SURGERY_ROOMS)
    
    queue_lengths = []    
    current_time = start_treatment_datetime
    last_treatment_end_time = start_treatment_datetime
    
    end_datetime = start_treatment_datetime + timedelta(days=SIM_DURATION_DAYS)
    
    min_patients_per_hour = 1
    max_patients_per_hour = 5
    
    patient_arrivals = generate_arrival_times(start_treatment_datetime, end_datetime, min_patients_per_hour, max_patients_per_hour, TIME_SLOT_MIN)
    
    for arrival_datetime in patient_arrivals: 
        patient_id += 1
        
        patient = Patient(patient_id)
        patient.set_priority()
        procedure_type = patient.set_patient_outcome()

        patients_in_queue += 1
        if procedure_type != 'Discharge':
            hospital.queue_counts[procedure_type] += 1

        print(f"Patient {patient_id} added to queue. Patients in queue: {patients_in_queue}")
        patient_process = env.process(assign_room(env, arrival_datetime, hospital, patient))
        env.run(until=patient_process)
    
        # Track queue lengths
        queue_lengths.append({
            'time': current_time.strftime('%H:%M:%S'),
            'queue_length': patients_in_queue
        })
    
        summary_data.append({
            'date': current_time.strftime('%Y-%m-%d'),
            'time_slot': current_time.strftime('%H:%M:%S'),
            'total_patients_in_queue': patients_in_queue,
            'patients_in_general_queue': hospital.queue_counts['General'],
            'patients_in_observation_queue': hospital.queue_counts['Observation'],
            'patients_in_surgery_queue': hospital.queue_counts['Surgery'],
            'patients_discharged': hospital.queue_counts['Discharge']
        })

        # Calculate summary metrics
        avg_wait_time = np.mean([patient['wait_time'] for patient in patient_data])
        avg_treatment_duration = np.mean([patient['duration'] for patient in patient_data])
        total_patients_processed = len(patient_data)

        summary_metrics_entry = {
            'date': current_time.strftime('%Y-%m-%d'),
            'time_slot': current_time.strftime('%H:%M:%S'),
            'avg_wait_time': avg_wait_time,
            'avg_treatment_duration': avg_treatment_duration,
            'total_patients_processed': total_patients_processed
        }
        summary_metrics_data.append(summary_metrics_entry)
        current_time += timedelta(minutes=TIME_SLOT_MIN)

    queue_lengths_df = pd.DataFrame(queue_lengths)
    arrival_times_df = pd.DataFrame(arrival_times, columns=['arrival_time'])
    room_usages_df = {key: pd.DataFrame(times, columns=['usage_time']) for key, times in room_usages.items()}

    return pd.DataFrame(summary_data), queue_lengths_df, arrival_times_df, room_usages_df

def get_summary_df():
    global summary_data
    return pd.DataFrame(summary_data)

def get_summary_metrics_df():
    global summary_metrics_data
    return pd.DataFrame(summary_metrics_data)

def get_patients_df():
    global patient_data
    return pd.DataFrame(patient_data)

def plot_patient_outcomes():
    global patient_data
    outcomes = [patient['procedure_type'] for patient in patient_data]
    outcome_counts = pd.Series(outcomes).value_counts(normalize=True) * 100

    fig = go.Figure(data=[go.Pie(labels=outcome_counts.index, values=outcome_counts, hole=0.3,
                                 textinfo='label+percent', textfont_size=14, marker=dict(colors=['#ff9999','#66b3ff','#99ff99']),
                                 pull=[0.1]*len(outcome_counts))])

    fig.update_layout(title_text='Proportion of Patient Outcomes', title_font_size=20,
                      legend_title_text='Outcome Type', legend_title_font_size=14,
                      legend_font_size=12, legend_font_color='white')

    st.plotly_chart(fig)


def plot_transition_probabilities():
    global transition_counts
    
    # Prepare data for bar chart
    data = []
    labels = ['General to Observation', 'General to Surgery', 'Observation to General', 'Observation to Surgery']
    for source in ['General', 'Observation']:
        for target in ['Observation', 'Surgery']:
            if source != target:
                count = transition_counts[source][target]
                data.append({'Transition': f"{source} to {target}", 'Count': count})

    df = pd.DataFrame(data)
    
    fig = px.bar(df, x='Transition', y='Count', title='Patient Procedure Transitions',
                 labels={'Count': 'Number of Transitions'},
                 color='Count',
                 color_continuous_scale='Blues')
    fig.update_layout(title_text='Patient Procedure Transitions', title_font_size=20)
    st.plotly_chart(fig)
    
def plot_queue_summary():
    global summary_data
    
    df = pd.DataFrame(summary_data)
    
    fig = px.line(df, x='time_slot', y=['patients_in_general_queue', 'patients_in_observation_queue', 'patients_in_surgery_queue'],
                  title='Number of Patients in Each Queue Over Time',
                  labels={'time_slot': 'Time Slot', 'value': 'Number of Patients'},
                  line_shape='linear')
    fig.update_layout(title_text='Patients in Queue by Procedure Types Over Time', title_font_size=20)
    st.plotly_chart(fig)

def plot_queue_length_summary(queue_lengths_df):
    # Calculate average and maximum queue lengths
    avg_queue_length = queue_lengths_df['queue_length'].mean()
    max_queue_length = queue_lengths_df['queue_length'].max()
    
    # Display metrics
    st.write(f"Average Queue Length: {avg_queue_length:.2f}")
    st.write(f"Maximum Queue Length: {max_queue_length}")

    # Plot average and max queue length
    fig = go.Figure()

    # Plot average queue length as a horizontal line
    fig.add_trace(go.Scatter(x=queue_lengths_df['time'], y=[avg_queue_length]*len(queue_lengths_df),
                             mode='lines', name='Average Queue Length', line=dict(color='blue', width=2)))
    
    # Plot queue length over time
    fig.add_trace(go.Scatter(x=queue_lengths_df['time'], y=queue_lengths_df['queue_length'],
                             mode='lines', name='Queue Length', line=dict(color='red', width=1)))

    fig.update_layout(title='Queue Length Over Time',
                      xaxis_title='Time',
                      yaxis_title='Queue Length',
                      legend_title='Legend')
    
    st.plotly_chart(fig)

    
def plot_summary_metrics():
    global summary_metrics_data
    
    df = pd.DataFrame(summary_metrics_data)
    
    fig = px.bar(df, x='date', y=['avg_wait_time', 'avg_treatment_duration'],
                 title='Average Wait Time and Treatment Duration',
                 labels={'value': 'Minutes'},
                 barmode='group')
    fig.update_layout(title_text='Average Wait Time and Treatment Duration', title_font_size=20)
    st.plotly_chart(fig)

def plot_heatmap(arrival_times_df):
    # Extract hour and weekday from arrival times
    arrival_times_df['hour'] = arrival_times_df['arrival_time'].dt.hour
    arrival_times_df['weekday'] = arrival_times_df['arrival_time'].dt.day_name()

    # Create pivot table for arrivals heatmap
    arrivals_pivot = arrival_times_df.pivot_table(index='weekday', columns='hour', aggfunc='size', fill_value=0)

    # Create heatmap for patient arrivals
    fig_arrivals = px.imshow(arrivals_pivot, labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Number of Arrivals'}, 
                             title='Heatmap of Patient Arrivals')

    st.plotly_chart(fig_arrivals)
    for room_type, usage_df in room_usages_df.items():
        usage_df['hour'] = usage_df['usage_time'].dt.hour
        usage_df['weekday'] = usage_df['usage_time'].dt.day_name()

        # Create pivot table for room usage heatmap
        usage_pivot = usage_df.pivot_table(index='weekday', columns='hour', aggfunc='size', fill_value=0)

        # Create heatmap for room usage
        fig_usage = px.imshow(usage_pivot, labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': f'Number of {room_type} Room Usages'}, 
                                title=f'Heatmap of {room_type} Room Usage')

        st.plotly_chart(fig_usage)

    
# Streamlit output
# run_simulation()
if st.sidebar.button("Run Simulation"):
    summary_df, queue_lengths_df, arrival_times_df, room_usages_df = run_simulation()
    
    patients_df = get_patients_df()
    st.write("Patients Arrivals Data", patients_df)
    
    plot_heatmap(arrival_times_df)
    
    plot_patient_outcomes()
        
    plot_transition_probabilities()

    summary_df = get_summary_df()
    st.write("Total Patients in Queue in Time Slots - Summary Data", summary_df)
    
    plot_queue_summary()

    summary_metrics_df = get_summary_metrics_df()
    st.write("Summary Metrics Data", summary_metrics_df)
    
    plot_summary_metrics()
    
    
    


