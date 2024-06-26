import simpy
import random
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import numpy as np
import faker as Faker
import uuid
from faker import Faker
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import queue
from queue import PriorityQueue
import itertools


# Constants
NUM_GENERAL_ROOMS = 1
NUM_SURGERY_ROOMS = 1
NUM_OBSERVATION_ROOMS = 1
NUM_ROOMS = 3

discharged_patients = [] 
summary_data = []
patient_data = []

patients_in_queue = 0
num_patients_treated = 0
room_availability = "None"

available_rooms = 0
general_room_occupancy = 0
surgery_room_occupancy = 0
observation_room_occupancy = 0


num_in_general_rooms = 0
num_in_surgery_rooms = 0
num_in_observation_rooms = 0

num_discharged = 0

general_room_next_free = 0
surgery_room_next_free = 0
observation_room_next_free = 0

# Track statistics
total_wait_time_general = 0
total_wait_time_surgery = 0
total_wait_time_observation = 0
total_patients_general = 0
total_patients_surgery = 0
total_patients_observation = 0


prob_observation_after_general = 0.5
prob_stay_general_after_general = 0.2
prob_surgery_after_general = 0.2
prob_discharge_after_general = 0.1

prob_surgery_after_observation = 0.5
prob_discharge_after_observation = 0.5

general_room_queue = PriorityQueue()
observation_room_queue = PriorityQueue()
surgery_room_queue = PriorityQueue()


# Initialize Faker
fake = Faker()

def generate_staff_identity_data(num_doctors, num_nurses_general, num_nurses_surgery, num_nurses_observation, doctor_shifts, nurse_shifts):
    staff_identity_data = []
    # Function to generate random age between 25 and 65
    def generate_age():
        return np.random.randint(25, 66)

    # Function to generate random years of experience between 1 and 30
    def generate_experience():
        return np.random.randint(1, 31)

    # Function to generate random cost per hour between 50 and 200
    def generate_cost_per_hour():
        return round(np.random.uniform(50, 200), 2)

    # Generate data for doctors
    for _ in range(num_doctors):
        staff_id = str(uuid.uuid4())
        staff_name = fake.name()
        age = generate_age()
        years_of_experience = generate_experience()
        cost_per_hour = generate_cost_per_hour()
        shift = doctor_shifts
        role = "Doctor"
        staff_identity_data.append([staff_id, staff_name, role, age, years_of_experience, cost_per_hour, shift])

    # Generate data for nurses
    for _ in range(num_nurses_general):
        staff_id = str(uuid.uuid4())
        staff_name = fake.name()
        age = generate_age()
        years_of_experience = generate_experience()
        cost_per_hour = generate_cost_per_hour()
        shift = nurse_shifts
        role = "Nurse"
        staff_identity_data.append([staff_id, staff_name, role, age, years_of_experience, cost_per_hour, shift])
        
    for _ in range(num_nurses_surgery):
        staff_id = str(uuid.uuid4())
        staff_name = fake.name()
        age = generate_age()
        years_of_experience = generate_experience()
        cost_per_hour = generate_cost_per_hour()
        shift = nurse_shifts
        role = "Nurse"
        staff_identity_data.append([staff_id, staff_name, role, age, years_of_experience, cost_per_hour, shift])
    
    for _ in range(num_nurses_observation):
        staff_id = str(uuid.uuid4())
        staff_name = fake.name()
        age = generate_age()
        years_of_experience = generate_experience()
        cost_per_hour = generate_cost_per_hour()
        shift = nurse_shifts
        role = "Nurse"
        staff_identity_data.append([staff_id, staff_name, role, age, years_of_experience, cost_per_hour, shift])

    # Create DataFrame
    staff_identity_df = pd.DataFrame(staff_identity_data, columns=["staff_id", "staff_name", "role", "age", "years_of_experience", "cost_per_hour", "shifts"])
    
    return staff_identity_df


def simulate_staff_schedule(num_days, start_datetime):
    # Initialize schedule dictionary
    schedule = []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Generate staff identity data
    staff_identity_df = generate_staff_identity_data(num_doctors, num_nurses_general, num_nurses_surgery, num_nurses_observation, doctor_shifts, nurse_shifts)
    # Generate schedule for each weekday
    for days in range(num_days):
        current_date = start_datetime + timedelta(days=days)

        if current_date.strftime("%A") in weekdays:
            for procedure_type in ["general", "surgery", "observation", "none"]:    
                # Assign doctors and nurses randomly to each procedure
                doctors = staff_identity_df[staff_identity_df['role'] == 'Doctor']
                nurses = staff_identity_df[staff_identity_df['role'] == 'Nurse']

                assigned_doctor = doctors.sample(1).iloc[0]
                if procedure_type == "surgery":
                    num_nurses = num_nurses_surgery
                    room_type = 'surgery'
                elif procedure_type == "observation":
                    num_nurses = num_nurses_observation 
                    room_type = 'observation'
                elif procedure_type == "general":
                    num_nurses = num_nurses_general
                    room_type = 'general'
                else:
                    room_type = 'none'
                
                assigned_nurses = nurses.sample(num_nurses)

                schedule.append({
                    "scheduled_date": current_date.strftime('%Y-%m-%d'),
                    "day": current_date.strftime("%A"),
                    "procedure_type": procedure_type,
                    "room_type": room_type,
                    "doctor_id": assigned_doctor['staff_id'],
                    "doctor_name": f"Dr. {assigned_doctor['staff_name']}",
                    "nurse_name(s)": [nurse['staff_name'] for _, nurse in assigned_nurses.iterrows()],
                    "doctor_shift_hours": assigned_doctor['shifts'],
                    "total_nurse_shift_hours": assigned_nurses['shifts'].sum()
                })

    staff_month_schedule = pd.DataFrame(schedule)
    staff_month_schedule.to_csv('staff_month_schedules.csv', index=False)
    return staff_month_schedule

class PatientAdmissions:
    def __init__(self, env):
        self.env = env
        self.general_room = simpy.Resource(env, capacity=NUM_GENERAL_ROOMS) #Room A
        self.observation_room = simpy.Resource(env,capacity=NUM_OBSERVATION_ROOMS) #Room B
        self.surgery_room = simpy.Resource(env, capacity=NUM_SURGERY_ROOMS) #Room C
        
    def general_procedure(self, patient_id):
        yield self.env.timeout(GENERAL_DURATION)

    def surgery_procedure(self, patient_id):
        yield self.env.timeout(SURGERY_DURATION)
    
    def observation_procedure(self, patient_id):
        yield self.env.timeout(OBSERVATION_DURATION)


def log_patient_data(patient_id, arrival_datetime, patient_status, room_type, procedure_type, treatment_start_time, treatment_end_time, wait_time, duration, log_type):
    global patient_data
    log_entry = {
        'patient_id': patient_id,
        'arrival_time': arrival_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'patient_status': patient_status,
        'room_type': room_type,
        'procedure_type': procedure_type,
        'treatment_start_time': treatment_start_time.strftime('%Y-%m-%d %H:%M:%S') if treatment_start_time != 'N/A' else 'N/A',
        'treatment_end_time': treatment_end_time.strftime('%Y-%m-%d %H:%M:%S') if treatment_end_time != 'N/A' else 'N/A',
        'wait_time': wait_time,
        'duration': duration
    }

    if log_type == 'Priority 1' or log_type == 'Priority 2':
        patient_data.append(log_entry)
    elif log_type == 'discharge':
        discharged_patients.append(log_entry)
    
def patient_admissions(env, start_datetime, arrival_time_minutes, hospital):
    
    global general_room_next_free, surgery_room_next_free, observation_room_next_free, patient_data, patients_in_queue, patient_id
    
    global general_room_queue, observation_room_queue, surgery_room_queue, patient_data, patient_id_gen, discharged_patients
    
    
    patient_id = 0

    # Calculate the arrival datetime
    arrival_datetime = start_datetime + timedelta(minutes=arrival_time_minutes % (24 * 60))
    # log_patient_data(patient_id, arrival_datetime, 'arrived', 'None', 'arrival', arrival_datetime, 'N/A', 0, 0)
    
    delay = random.randint(30, 60)
    yield env.timeout(delay)
    
    treatment_start_time = arrival_datetime + timedelta(minutes=delay)
    wait_time = (treatment_start_time - arrival_datetime).total_seconds() / 60  # Calculate the wait time

    patient_status = random.choices(['Priority 1', 'Priority 2', 'to discharge'], weights=[0.3, 0.4, 0.3], k=1)[0]

    # Priority 1: direct to C (surgery)
    # Priority 2:
    # Priority 3:
    # emergency Room A => Room C / Room C => B # Priority 1
    # direct admission B (observation) => C (surgery) => back to room B #Priority 2
    # planned/elective - Room A => Room B  #Priority 3
    
    if patient_status == 'Priority 1':
        general_room_queue.put((1, patient_id))  # Highest priority for surgery
    elif patient_status == 'Priority 2':
        general_room_queue.put((2, patient_id))  # Regular priority for general/observation
    else:
        discharged_patients.append(patient_id)  # Directly log to discharge patients
        log_patient_data(patient_id, arrival_datetime, patient_status, 'None', 'None', arrival_datetime, 'N/A', 0, 0, log_type='discharge')
        return 
    
    procedure_type = random.choice(['general', 'observation', 'surgery'])
    room_type = 'N/A'
    

    if procedure_type == 'general':
        room_type = 'Room A'
        duration = GENERAL_DURATION
        if env.now < general_room_next_free:
            yield env.timeout(max(0, general_room_next_free - env.now))
        general_room_next_free = env.now + duration
        yield env.timeout(duration)
    elif procedure_type == 'observation':
        room_type = 'Room B'
        if env.now < observation_room_next_free:
            yield env.timeout(max(0, observation_room_next_free - env.now))
        duration = OBSERVATION_DURATION
        observation_room_next_free = env.now + duration
        yield env.timeout(duration)
    elif procedure_type == 'surgery':
        
        room_type = 'Room C'
        if env.now < surgery_room_next_free:
            yield env.timeout(max(0, surgery_room_next_free - env.now))
        duration = SURGERY_DURATION
        surgery_room_next_free = env.now + duration
    
        yield env.timeout(duration)
    
    patient_id +=1 

    # # Main patient process
    # while True:
    #     # General Room Assignment
    #     if hospital.general_room.count < hospital.general_room.capacity and not general_room_queue.empty():
    #         priority = general_room_queue.get()
    #         patient_id, wait_time, duration = yield env.process(handle_patient(env, start_datetime, arrival_datetime, patient_id, hospital, patient_status, 'general', room_type, wait_time, duration))

    #     # Observation Room Assignment
    #     if hospital.observation_room.count < hospital.observation_room.capacity and not observation_room_queue.empty():
    #         priority = observation_room_queue.get()
    #         patient_id, wait_time, duration = yield env.process(handle_patient(env, start_datetime, arrival_datetime, patient_id, hospital, patient_status, 'observation', room_type, wait_time, duration))

    #     # Surgery Room Assignment
    #     if hospital.surgery_room.count < hospital.surgery_room.capacity and not surgery_room_queue.empty():
    #         priority = surgery_room_queue.get()
    #         patient_id, wait_time, duration = yield env.process(handle_patient(env, start_datetime, arrival_datetime, patient_id, hospital, patient_status, 'surgery', room_type, wait_time, duration))

    #     yield env.timeout(random.randint(10, 20))  # Adjust arrival time interval
        
    yield env.process(handle_patient(env, start_datetime, arrival_datetime, patient_id, hospital, patient_status, procedure_type, room_type, wait_time, duration))

        

def handle_patient(env, start_datetime, arrival_datetime, patient_id, hospital, patient_status, procedure_type, room_type, wait_time, duration):

        global general_room_next_free, surgery_room_next_free, observation_room_next_free

        # Randomize delay between arrival and treatment start time (30 minutes to 1 hour)
        delay = random.randint(30, 60)
        yield env.timeout(delay)
        treatment_start_time = arrival_datetime + timedelta(minutes=delay) 
        
        treatment_end_time = treatment_start_time + timedelta(minutes=duration)

        with getattr(hospital, f"{procedure_type}_room").request() as request:
            yield request
            print(f"{patient_id} starts {procedure_type} at {treatment_start_time}")
            yield env.process(getattr(hospital, f"{procedure_type}_procedure")(patient_id))
            print(f"{patient_id} finishes {procedure_type} at {treatment_end_time}")
            
            log_patient_data(patient_id, arrival_datetime, 'Priority 2', procedure_type, room_type, treatment_start_time, treatment_end_time, wait_time, duration, 'Priority 2')

            # Determine next steps after each procedure
            if procedure_type == 'general':
                rand_num = random.random()
                if rand_num < prob_surgery_after_general:
                    surgery_room_queue.put((1, patient_id))  # Highest priority for surgery
                    procedure_type = 'surgery'
                    room_type = 'Room C'
                    log_patient_data(patient_id, arrival_datetime, 'Priority 1', procedure_type, room_type, treatment_start_time, treatment_end_time, wait_time, duration, 'Priority 1')
                elif rand_num < prob_surgery_after_general + prob_observation_after_general:
                    observation_room_queue.put((2, patient_id))  # Priority for observation/testing
                    procedure_type = 'observation'
                    room_type = 'Room B'
                elif rand_num < prob_surgery_after_general + prob_observation_after_general + prob_stay_general_after_general:
                    general_room_queue.put((2, patient_id))  # Priority to stay in general room
                else:
                    log_patient_data(patient_id, arrival_datetime, 'to discharge', 'None', 'None', treatment_start_time, 'N/A', wait_time, 0, 'discharge')

            elif procedure_type == 'observation':
                # general_end_time = arrival_datetime + timedelta(minutes=delay + GENERAL_DURATION)
                
                # if treatment_start_time < general_end_time:
                #     yield env.timeout((general_end_time - treatment_start_time).total_seconds() / 60)
                #     treatment_start_time = general_end_time

                if random.random() < prob_surgery_after_observation:
                    surgery_room_queue.put((1, patient_id))  # Priority for immediate surgery
                    log_patient_data(patient_id, arrival_datetime, 'Priority 1', 'surgery', 'Room C', treatment_start_time, treatment_end_time, wait_time, duration, 'Priority 1')
                else:
                    log_patient_data(patient_id, arrival_datetime, 'discharge', 'None', 'None', treatment_start_time, 'N/A', wait_time, 0, 'discharge')
            
            elif procedure_type == 'surgery':
                # Ensure surgery starts after general screening ends
                # general_end_time = arrival_datetime + timedelta(minutes=delay + GENERAL_DURATION)
                # if treatment_start_time < general_end_time:
                #     yield env.timeout((general_end_time - treatment_start_time).total_seconds() / 60)
                #     treatment_start_time = general_end_time
                log_patient_data(patient_id, arrival_datetime, patient_status, 'discharge', procedure_type, treatment_start_time, treatment_end_time, wait_time, duration, 'discharge')

            return patient_id, wait_time, duration

def get_patients_df():
    global patient_data
    patient_df = pd.DataFrame(patient_data)
    return patient_df


def simulate_hospital(sim_duration_days, start_datetime, time_slot_min):
    global patient_data
    env = simpy.Environment()
    current_time = start_datetime
    for day in range(sim_duration_days):
        for time_slot in range(0, 24 * 60, time_slot_min):
            arrival_time_minutes = (day * 24 * 60) + time_slot + random.randint(0, time_slot_min - 1)
            for _ in range(random.randint(1, 5)):
                hospital = PatientAdmissions(env)
                env.process(patient_admissions(env, current_time, arrival_time_minutes, hospital))
                current_time = start_datetime + timedelta(minutes=arrival_time_minutes)
    env.run(until=sim_duration_days * 24 * 60)
    # patients_df = get_patients_df()
    # print("Patients DataFrame", patients_df)

# Calculate averages    
def calculate_averages():
    global total_wait_time_general, total_wait_time_surgery, total_patients_general, total_patients_surgery
    avg_wait_time_general = total_wait_time_general / total_patients_general if total_patients_general > 0 else 0
    avg_wait_time_surgery = total_wait_time_surgery / total_patients_surgery if total_patients_surgery > 0 else 0
    return avg_wait_time_general, avg_wait_time_surgery
   

def patient_arrivals(env, start_datetime, time_slot_min, sim_duration_days):
    global patients_in_queue, total_wait_time_general, total_wait_time_surgery, patient_id
    global total_patients_general, total_patients_surgery, summary_data
    global general_room_next_free, surgery_room_next_free, general_room_occupancy, surgery_room_occupancy, observation_room_occupancy
    global total_wait_time_observation, total_patients_observation, num_discharged
    
    for day in range(sim_duration_days):
        for time_slot in range(0, 24 * 60, time_slot_min):
            num_new_patients_this_slot = 0
            arrival_time_minutes = (day * 24 * 60) + time_slot + random.randint(0, time_slot_min - 1)
            for _ in range(random.randint(1, 5)): # Adjust the range for random arrival count per slot
                hospital = PatientAdmissions(env)
                patient_admissions_result = yield env.process(patient_admissions(env, start_datetime, arrival_time_minutes, hospital))
                priority, patient_id, wait_time, duration = patient_admissions_result
                num_new_patients_this_slot += 1
                
                patients_in_queue += num_new_patients_this_slot   

                if duration == GENERAL_DURATION:
                    total_wait_time_general += wait_time
                    total_patients_general += 1
                    general_queue_size = priority.qsize()

                    
                elif duration == OBSERVATION_DURATION:
                    total_wait_time_observation += wait_time
                    total_patients_observation += 1
                    observation_queue_size = priority.qsize()
                    
                elif duration == SURGERY_DURATION:
                    total_wait_time_surgery += wait_time
                    total_patients_surgery += 1
                    surgery_queue_size = priority.qsize()
                
                elif duration == 0:
                    num_discharged += 1
                    
                available_general_rooms = NUM_GENERAL_ROOMS - general_room_occupancy
                available_observation_rooms = NUM_OBSERVATION_ROOMS - observation_room_occupancy
                available_surgery_rooms = NUM_SURGERY_ROOMS - surgery_room_occupancy
                
                if env.now >= general_room_next_free and general_room_occupancy < NUM_GENERAL_ROOMS:
                    general_room_occupancy += 1
                    available_general_rooms -= 1
                    if patients_in_queue > 0:
                        patients_in_queue -= 1
                        
                if env.now >= observation_room_next_free and observation_room_occupancy < NUM_OBSERVATION_ROOMS:
                    observation_room_occupancy += 1
                    available_observation_rooms -= 1
                    if patients_in_queue > 0:
                        patients_in_queue -= 1

                if env.now >= surgery_room_next_free and surgery_room_occupancy < NUM_SURGERY_ROOMS:
                    surgery_room_occupancy += 1
                    available_surgery_rooms -= 1
                    if patients_in_queue > 0:
                        patients_in_queue -= 1
                
            
                current_datetime = start_datetime + timedelta(minutes=arrival_time_minutes)

                summary_data.append({
                    'Date': current_datetime.strftime('%Y-%m-%d'),
                    'Time Slot': current_datetime.strftime('%H:%M:%S'),
                    'New Patients': num_new_patients_this_slot,
                    'Patients in Queue': patients_in_queue,
                    'Patients in Observation Room': observation_queue_size,
                    'Patients in Surgery Room': surgery_queue_size,
                    'Patients in General Room': general_queue_size
                }, ignore_index=True)
                

    return summary_data
                         
def get_summary_df():
    patient_summary_df= pd.DataFrame(summary_data)
    return patient_summary_df  

# Streamlit UI
st.title("Hospital Operations Scheduling Simulation")

#Select number of doctors and nurses
num_doctors = st.sidebar.slider("Number of Doctors", min_value=1, max_value=6, value=3)
num_nurses_general = st.sidebar.slider("Number of Nurses (General)", min_value=1, max_value=4, value=2)
num_nurses_surgery =st.sidebar.slider("Number of Nurses (Surgery)", min_value=2, max_value=8, value=4)
num_nurses_observation  = st.sidebar.slider("Number of Nurses (Observation)", min_value=2, max_value=8, value=4)

#Set mean duration of procedures general/surgery 
GENERAL_DURATION = st.sidebar.slider("Mean Duration of General Consultation", min_value=30, max_value=60, value=45)
SURGERY_DURATION = st.sidebar.slider("Mean Duration of Surgery Procedure", min_value=0, max_value=120, value=90)
OBSERVATION_DURATION =  st.sidebar.slider("Mean Duration of Testing/Observation Procedure", min_value=0, max_value=90, value=60)

NUM_GENERAL_ROOMS = st.sidebar.slider("Number of General Rooms", min_value=1, max_value=10, value=2)
NUM_SURGERY_ROOMS = st.sidebar.slider("Number of Surgery Rooms", min_value=1, max_value=10, value=2)
NUM_OBSERVATION_ROOMS = st.sidebar.slider("Number of Observation Rooms", min_value=1, max_value=10, value=2)

sim_duration_days = st.sidebar.slider("Simulation Duration (Days)", min_value=1, max_value=31, value=7)

#Select time slot min
time_slot_min  = st.sidebar.slider("Time Slot Intervals (in min)", min_value=0, max_value=60, value=15)

# start_date = st.date_input("Start Date", min_value=datetime(2024, 5, 1), max_value=datetime(2024, 5, 31), value=datetime(2024, 5, 2))
doctor_shifts = st.sidebar.slider("Doctor Shift (in hours)", min_value=3, max_value=8, value=5)
nurse_shifts = st.sidebar.slider("Nurse Shift (in hours)", min_value=7, max_value=14, value=8)

# start_datetime = datetime.now()
start_datetime = datetime(2024, 6, 1)

simulate_hospital(sim_duration_days, start_datetime, time_slot_min)

# Simulate button
if st.sidebar.button("Simulate"):
    staff_month_schedule = simulate_staff_schedule(sim_duration_days, start_datetime)
    st.write("Doctors/Nurses Staff Schedule:")
    st.write(staff_month_schedule)

    # avg_wait_time_general, avg_wait_time_surgery = calculate_averages()

    # st.write(f"Average Wait Time for General Checkup: {avg_wait_time_general:.2f} minutes")
    # st.write(f"Average Wait Time for Surgery: {avg_wait_time_surgery:.2f} minutes")
    
    # Plot Average Wait Time for General Checkup and Surgery
    # wait_times = {'General Checkup': avg_wait_time_general, 'Surgery': avg_wait_time_surgery}
    # fig, ax = plt.subplots()
    # ax.bar(wait_times.keys(), wait_times.values())
    # ax.set_title('Average Wait Time')
    # ax.set_ylabel('Minutes')
    # st.pyplot(fig)
    
    st.write("Total Patients in Queue/Procedures Summary Schedule")
    patients_summary_df = get_summary_df()
    st.write(patients_summary_df)
    
    #Load patient data
    patients_df = get_patients_df()
    st.write("Patients Arrivals Table")
    st.write(patients_df)


    # # Merge patient data with staff schedule based on procedure type
    # merge_patients_staff_df = pd.merge(staff_month_schedule, patients_df, on='procedure_type', how='inner')

    # # Calculate average wait time per doctor
    # avg_wait_time_per_doctor = merge_patients_staff_df.groupby('doctor_name')['wait_time'].mean()
    # st.write("Average Wait Time per Doctor:")
    # st.write(avg_wait_time_per_doctor)
    
    # # Plot Average Wait Time per Doctor
    # fig, ax = plt.subplots()
    # avg_wait_time_per_doctor.plot(kind='bar', ax=ax)
    # ax.set_title('Average Wait Time per Doctor')
    # ax.set_ylabel('Minutes')
    # st.pyplot(fig)

    


    

    
   


