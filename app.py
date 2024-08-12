import simpy
import random
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import uuid
from faker import Faker

# Constants
#Select number of doctors and nurses
num_doctors = st.sidebar.slider("Number of Doctors", min_value=1, max_value=50, value=10)
num_nurses_general = st.sidebar.slider("Number of Nurses (General)", min_value=1, max_value=20, value=10)
num_nurses_surgery =st.sidebar.slider("Number of Nurses (Surgery)", min_value=2, max_value=20, value=10)
num_nurses_observation  = st.sidebar.slider("Number of Nurses (Observation)", min_value=1, max_value=20, value=10)

GENERAL_DURATION = st.sidebar.slider("Mean Duration of General Consultation", min_value=30, max_value=60, value=45)
SURGERY_DURATION = st.sidebar.slider("Mean Duration of Surgery Procedure", min_value=0, max_value=120, value=90)
OBSERVATION_DURATION = st.sidebar.slider("Mean Duration of Testing/Observation Procedure", min_value=0, max_value=90, value=60)
NUM_GENERAL_ROOMS = st.sidebar.slider("Number of General Rooms", min_value=1, max_value=10, value=2)
NUM_SURGERY_ROOMS = st.sidebar.slider("Number of Surgery Rooms", min_value=1, max_value=10, value=2)
NUM_OBSERVATION_ROOMS = st.sidebar.slider("Number of Observation Rooms", min_value=1, max_value=10, value=2)
TIME_SLOT_MIN = st.sidebar.slider("Time Slot Intervals (in min)", min_value=0, max_value=60, value=15)
SIM_DURATION_DAYS = st.sidebar.slider("Number of Days", min_value=0, max_value=31, value=7)

start_treatment_datetime = datetime.now()


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

# Initialize Faker
fake = Faker()

# Generate random date within simulation duration
def random_date_within_sim_duration(start_date, duration_days):
    delta_days = random.randint(0, duration_days - 1)
    return start_date + timedelta(days=delta_days)

def random_shift_times():
    # Define shift hour ranges (start, end) with a minimum of 4 hours duration
    shift_ranges = {
        "early morning": (0, 4, 7, 12),   #Starts between 4-6AM, ends at 9AM-12PM
        "morning": (7, 10, 11, 14),       # Start between 7-10 AM, end between 11 AM - 2 PM
        "afternoon": (12, 14, 16, 18),    # Start between 12-2 PM, end between 4-6 PM
        "evening": (17, 19, 21, 23),      # Start between 5-7 PM, end between 9-11 PM
        "late_night": (20, 23, 0, 4)      # Start between 8-11 PM, end between 3-5 AM
    }
    
    # Randomly select a shift type
    shift_type = random.choice(list(shift_ranges.keys()))
    
    # Get the start and end hour ranges for the selected shift
    start_hour_range = shift_ranges[shift_type][:2]
    end_hour_range = shift_ranges[shift_type][2:]
    
    # Randomly select a start hour within the start range
    start_hour = random.randint(start_hour_range[0], start_hour_range[1])
    
    # Calculate the earliest valid end hour (minimum 4 hours after start)
    earliest_end_hour = (start_hour + 4) % 24  # Use modulo 24 to handle overflow
    
    # Adjust end_hour_range to prevent invalid selections
    if earliest_end_hour > start_hour:
        # Same day shift
        latest_end_hour = min(end_hour_range[1], 23)
    else:
        # Shift crossing midnight
        if end_hour_range[1] < earliest_end_hour:
            latest_end_hour = end_hour_range[1] + 24
        else:
            latest_end_hour = end_hour_range[1]
    
    # Randomly select an end hour within the valid end range
    if earliest_end_hour <= latest_end_hour:
        end_hour = random.randint(earliest_end_hour, latest_end_hour) % 24
    else:
        # Handle crossing midnight with two possible ranges
        if random.random() < 0.5:
            end_hour = random.randint(earliest_end_hour, 23)
        else:
            end_hour = random.randint(0, latest_end_hour % 24)
    
    # Convert start_hour and end_hour to time format
    shift_start_time = datetime.strptime(f"{start_hour:02d}:00", '%H:%M').time()
    shift_end_time = datetime.strptime(f"{end_hour:02d}:00", '%H:%M').time()
    
    if end_hour < start_hour:
        shift_end_time = (datetime.combine(datetime.today(), shift_end_time) + timedelta(days=1)).time()
        
    shifts = []
    for shift_type, (start_min, start_max, end_min, end_max) in shift_ranges.items():
        for start_hour in range(start_min, start_max + 1):
            for end_hour in range(end_min, end_max + 1):
                # Ensure end hour is at least 4 hours after start
                if end_hour >= start_hour + 4:
                    if shift_type == 'late_night' and end_hour < start_hour:
                        end_hour += 24
                    shifts.append((start_hour, end_hour, shift_type))
    

    return shift_start_time, shift_end_time

def generate_schedule(num_doctors, num_nurses_general, num_nurses_surgery, num_nurses_observation, start_date, duration_days):
    doctors = []
    nurses = []
    
    procedure_types = ['General', 'Surgery', 'Observation']
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Generate doctors
    for doctor_id in range(1, num_doctors + 1):
        doctor_name = fake.name()
        procedure_type = random.choice(procedure_types)
        # availability_date = random_date_within_sim_duration(start_date, duration_days)
        for day in range(duration_days):
            shift_start, shift_end = random_shift_times()
            availability_date = start_date + timedelta(days=day)
            if availability_date.strftime("%A") in weekdays:
                doctors.append({
                    'doctor_id': doctor_id,
                    'doctor_name': doctor_name,
                    'procedure_type': procedure_type,
                    'availability_date': availability_date.strftime('%Y-%m-%d'),
                    'day': availability_date.strftime("%A"),
                    'shift_start': shift_start.strftime('%H:%M'),
                    'shift_end': shift_end.strftime('%H:%M')
                })
    
    # Generate nurses
    for nurse_id in range(1, num_nurses_general + num_nurses_surgery + num_nurses_observation + 1):
        nurse_name = fake.name()
        if nurse_id <= num_nurses_general:
            procedure_type = 'General'
        elif nurse_id <= num_nurses_general + num_nurses_surgery:
            procedure_type = 'Surgery'
        else:
            procedure_type = 'Observation'
        
        for day in range(duration_days):
            shift_start, shift_end = random_shift_times()
            # availability_date = random_date_within_sim_duration(start_date, duration_days)
            availability_date = start_date + timedelta(days=day)
            if availability_date.strftime("%A") in weekdays:
                nurses.append({
                    'nurse_id': nurse_id,
                    'nurse_name': nurse_name,
                    'procedure_type': procedure_type,
                    'availability_date': availability_date.strftime('%Y-%m-%d'),
                    'shift_start': shift_start.strftime('%H:%M'),
                    'shift_end': shift_end.strftime('%H:%M')
                })
    
    return doctors, nurses

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
        self.assigned_doctor = None
        self.assigned_nurses = []

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
    
def parse_time(time_str):
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M')

# Function to find available medical staff
def find_available_staff(arrival_time, procedure_type, doctors, nurses):
    assigned_doctor = None
    assigned_nurses = []

    # Find available doctor
    for doctor in doctors:
        if (doctor['procedure_type'] == procedure_type and
            doctor['availability_date'] == arrival_time.strftime('%Y-%m-%d') and
            doctor['shift_start'] <= arrival_time.strftime('%H:%M') <= doctor['shift_end']):
            assigned_doctor = doctor['doctor_name']
            break

    # Find available nurses
    for nurse in nurses:
        if (nurse['procedure_type'] == procedure_type and
            nurse['availability_date'] == arrival_time.strftime('%Y-%m-%d') and
            nurse['shift_start'] <= arrival_time.strftime('%H:%M') <= nurse['shift_end']):
            assigned_nurses.append(nurse['nurse_name'])

    return assigned_doctor, assigned_nurses

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
    
    # Assign medical staff
    assigned_doctor, assigned_nurses = find_available_staff(treatment_start_time, procedure_type, doctors, nurses)
    patient.assigned_doctor = assigned_doctor
    patient.assigned_nurses = assigned_nurses

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
    
    log_patient_data(patient.patient_id, arrival_datetime, priority, procedure_type, patient.assigned_doctor, patient.assigned_nurses, treatment_start_time, treatment_end_time, wait_time, duration)
    
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

def log_patient_data(patient_id, arrival_datetime, priority, procedure_type, assigned_doctor, assigned_nurse, treatment_start_time, treatment_end_time, wait_time, duration):
    global patient_data
    log_entry = {
        'patient_id': patient_id,
        'arrival_time': arrival_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'priority': priority,
        'procedure_type': procedure_type,
        'assigned_doctor': assigned_doctor,
        'assigned_nurse': assigned_nurse,
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
    st.title("Hospital Operations Scheduling Simulation")
    
    doctors, nurses = generate_schedule(
    num_doctors=num_doctors,
    num_nurses_general=num_nurses_general,
    num_nurses_surgery=num_nurses_surgery,
    num_nurses_observation=num_nurses_observation,
    start_date=start_treatment_datetime,
    duration_days=SIM_DURATION_DAYS
    )
    
    # shift_start_hour, shift_end_hour = random_shift_times()
    # print(f"Randomly selected shift: Start at {shift_start_hour}:00, End at {shift_end_hour}:00")
        
    st.write("Doctors Weekday Schedule:")
    st.dataframe(doctors)

    st.write("Nurses Weekday Schedule:")
    st.dataframe(nurses) 
    
    summary_df, queue_lengths_df, arrival_times_df, room_usages_df = run_simulation()

    patients_df = get_patients_df()
    st.write("Patients Arrivals Data", patients_df)
    
    summary_df = get_summary_df()
    st.write("Total Patients in Queue in Time Slots - Summary Data", summary_df)
    
    plot_heatmap(arrival_times_df)
    
    plot_patient_outcomes()
        
    plot_transition_probabilities()

  
    plot_queue_summary()

    summary_metrics_df = get_summary_metrics_df()
    st.write("Summary Metrics Data", summary_metrics_df)
    
    plot_summary_metrics()
    
    
    


