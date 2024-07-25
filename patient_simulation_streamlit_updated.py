import simpy
import random
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import numpy as np
import uuid
from faker import Faker
import matplotlib.pyplot as plt
from queue import PriorityQueue



# Constants
NUM_GENERAL_ROOMS = 1
NUM_SURGERY_ROOMS = 1
NUM_OBSERVATION_ROOMS = 1
NUM_ROOMS = 3

discharged_patients = [] 
summary_data = []
patient_data = []
total_patients_data = []

patients_in_queue = 0
num_patients_admitted = 0
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



class Hospital:
    def __init__(self, env, num_general_rooms, num_observation_rooms, num_surgery_rooms, num_waiting_rooms, patients_in_queue):
        self.env = env
        self.general_room = simpy.Resource(env, capacity=num_general_rooms)  # Room A
        self.observation_room = simpy.Resource(env, capacity=num_observation_rooms)  # Room B
        self.surgery_room =  simpy.Resource(env, capacity=num_surgery_rooms)  # Room C
        self.waiting_room = simpy.Resource(env, capacity=10) 
        self.patients_in_queue = patients_in_queue
        # self.num_doctors = num_doctors
        # self.num_nurse_observation = num_nurses_observation
        # self.num_nurse_general = num_nurses_general
        # self.num_nurses_surgery = num_nurses_surgery

class Patient:
    def __init__(self, env, patient_id, priority, procedure_type):
        self.env = env
        self.patient_id = patient_id
        self.priority = priority
        self.status = "waiting"
        self.procedure_type = procedure_type

def patient(env, start_datetime, arrival_time_minutes, hospital):
    
    global general_room_next_free, surgery_room_next_free, observation_room_next_free, patient_data, patient_id
    
    global general_room_queue, observation_room_queue, surgery_room_queue, patient_data, discharged_patients
    
    procedure_type = random.choice(['general', 'observation', 'surgery'])
    priority = random.choices(['Priority 1', 'Priority 2', 'discharge'], weights=[0.3, 0.4, 0.3], k=1)[0]
    patient_id = 0

    # Calculate the arrival datetime
    arrival_datetime = start_datetime + timedelta(minutes=arrival_time_minutes % (24 * 60))
    # log_patient_data(patient_id, arrival_datetime, 'arrived', 'None', 'arrival', arrival_datetime, 'N/A', 0, 0)
    
    delay = random.randint(30, 60)
    yield env.timeout(delay)
    treatment_start_time = arrival_datetime + timedelta(minutes=delay)
    print("Hello patient")
    patient_process = env.process(assign_room(env, hospital))
    
    env.run(until=patient_process)
    duration, room_type = patient_process.value
    print(duration, room_type)
    print("Here")
    
    treatment_end_time = treatment_start_time + timedelta(minutes=duration)

    hospital = Hospital(env, NUM_GENERAL_ROOMS, NUM_OBSERVATION_ROOMS, NUM_SURGERY_ROOMS, NUM_WAITING_ROOMS, patients_in_queue)

    with hospital.waiting_room.request() as req:
        yield req
        patient_id +=1 
        hospital.patients_in_queue += 1
        wait_time = (treatment_start_time - arrival_datetime).total_seconds() / 60  # Calculate the wait time
        print(f"{patient_id} enters waiting room at {env.now} after waiting {wait_time}")
        
        duration, room_type = yield env.process(assign_room(env, hospital))
        treatment_end_time = treatment_start_time + timedelta(minutes=duration)
        
        hospital.patients_in_queue -= 1
        log_patient_data(patient_id, arrival_datetime, priority, room_type, procedure_type, treatment_start_time, treatment_end_time, wait_time, duration)

        if procedure_type == 'general':
            rand_num = random.random()
            if rand_num < prob_surgery_after_general:
                procedure_type = 'surgery'
                print(f'Patient arrived. Patients in queue: {hospital.patients_in_queue}')
                log_total_patients(patients_in_queue)                
                log_patient_data(patient_id, arrival_datetime, 'Priority 1', procedure_type, room_type, treatment_start_time, treatment_end_time, wait_time, duration)
            elif rand_num < prob_surgery_after_general + prob_observation_after_general:
                procedure_type = 'observation'
                room_type = 'Room B'
                print(f'Patient arrived. Patients in queue: {hospital.patients_in_queue}')
                log_total_patients(patients_in_queue)
                log_patient_data(patient_id, arrival_datetime, 'Priority 2', procedure_type, room_type, treatment_start_time, treatment_end_time, wait_time, duration)
            elif rand_num < prob_surgery_after_general + prob_observation_after_general + prob_stay_general_after_general:
                procedure_type = 'general'
                room_type = 'Room A'
                print(f'Patient arrived. Patients in queue: {hospital.patients_in_queue}')
                log_total_patients(patients_in_queue)
                log_patient_data(patient_id, arrival_datetime, 'Priority 3', procedure_type, room_type, treatment_start_time, treatment_end_time, wait_time, duration)
            else:
                # discharged_patients.append(patient_id)
                # log_patient_data(patient_id, arrival_datetime, 'discharge', 'None', 'None', treatment_start_time, 'N/A', wait_time, 0)
                return

        elif procedure_type == 'observation':
            if random.random() < prob_surgery_after_observation:
                log_total_patients(patients_in_queue)
                log_patient_data(patient_id, arrival_datetime, 'Priority 1', 'surgery', 'Room C', treatment_start_time, treatment_end_time, wait_time, duration)
            else:
                # discharged_patients.append(patient_id)
                log_patient_data(patient_id, arrival_datetime, 'discharge', 'None', 'None', treatment_start_time, 'N/A', wait_time, 0)
        elif procedure_type == 'surgery':
            # discharged_patients.append(patient_id)
            # log_patient_data(patient_id, arrival_datetime, priority, 'discharge', procedure_type, treatment_start_time, treatment_end_time, wait_time, duration)
            return
        
    return patients_in_queue

def assign_room(env, hospital):
    duration = GENERAL_DURATION
    room_type = 'N/A'
    procedure_type = random.choice(['general', 'observation', 'surgery'])

    hospital = Hospital(env, NUM_GENERAL_ROOMS, NUM_OBSERVATION_ROOMS, NUM_SURGERY_ROOMS, NUM_WAITING_ROOMS, patients_in_queue)
    
    if procedure_type == 'general':
        room_resource = hospital.general_room
        room_type = 'General Room'
        next_free = general_room_next_free
        next_free_update = lambda now: now + GENERAL_DURATION
        next_free_var = 'general_room_next_free'
    elif procedure_type == 'observation':
        room_resource = hospital.observation_room
        room_type = 'Observation Room'
        duration = OBSERVATION_DURATION
        next_free = observation_room_next_free
        next_free_update = lambda now: now + OBSERVATION_DURATION
        next_free_var = 'observation_room_next_free'
    elif procedure_type == 'surgery':
        room_resource = hospital.surgery_room
        room_type = 'Surgery Room'
        duration = SURGERY_DURATION
        next_free = surgery_room_next_free
        next_free_update = lambda now: now + SURGERY_DURATION
        next_free_var = 'surgery_room_next_free'
    
    with room_resource.request() as request:
        yield request
        if env.now < next_free:
            yield env.timeout(next_free - env.now)
        setattr(hospital, next_free_var, next_free_update(env.now))
    
    return duration, room_type
       
def log_total_patients(patients_in_queue):
    global total_patients_data
    
    log_entry = {
        'patients_in_queue': patients_in_queue
    }
    total_patients_data.append(log_entry)
    
    total_patients_df = pd.DataFrame(total_patients_data)
    # print(f'Logged patients in queue: {patients_in_queue}')
    # print(total_patients_df)

    return total_patients_df

def log_patient_data(patient_id, arrival_datetime, priority, room_type, procedure_type, treatment_start_time, treatment_end_time, wait_time, duration):
    global patient_data
    log_entry = {
        'patient_id': patient_id,
        'arrival_time': arrival_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'priority': priority,
        'room_type': room_type,
        'procedure_type': procedure_type,
        'treatment_start_time': treatment_start_time.strftime('%Y-%m-%d %H:%M:%S') if treatment_start_time != 'N/A' else 'N/A',
        'treatment_end_time': treatment_end_time.strftime('%Y-%m-%d %H:%M:%S') if treatment_end_time != 'N/A' else 'N/A',
        'wait_time': wait_time,
        'duration': duration
    }
    patient_data.append(log_entry)
    return patient_data

def get_patients_df():
    global patient_data
    patient_df = pd.DataFrame(patient_data)
    return patient_df

def get_summary_df():
    global summary_data
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def simulate_hospital(sim_duration_days, start_datetime, time_slot_min):
    global total_patients_data, patient_data, summary_data, num_patients_admitted, patients_in_queue, general_room_occupancy, observation_room_occupancy, surgery_room_occupancy, patients_df
    env = simpy.Environment()
    current_time = start_datetime
    
    hospital = Hospital(env, NUM_GENERAL_ROOMS, NUM_OBSERVATION_ROOMS, NUM_SURGERY_ROOMS, NUM_WAITING_ROOMS, patients_in_queue)
    
    while (current_time - start_datetime).days < sim_duration_days:
    # for day in range(sim_duration_days):
        # for time_slot in range(0, 24 * 60, time_slot_min):
            # arrival_time_minutes = (day * 24 * 60) + time_slot + random.randint(0, time_slot_min - 1)
        arrival_time_minutes = random.randint(0, time_slot_min * 24)

        env.process(patient(env, current_time, arrival_time_minutes, hospital))     
       
        # total_patients_df = log_total_patients(patients_in_queue)
        # patients_in_queue = total_patients_df['patients_in_queue'].iloc[-1]
        # print("Total patients in queue", patients_in_queue)

        if env.now >= general_room_next_free and general_room_occupancy < NUM_GENERAL_ROOMS:
            general_room_occupancy += 1
            if patients_in_queue > 0:
                patients_in_queue -= 1

        if env.now >= observation_room_next_free and observation_room_occupancy < NUM_OBSERVATION_ROOMS:
            observation_room_occupancy += 1
            if patients_in_queue > 0:
                patients_in_queue -= 1

        if env.now >= surgery_room_next_free and surgery_room_occupancy < NUM_SURGERY_ROOMS:
            surgery_room_occupancy += 1
            if patients_in_queue > 0:
                patients_in_queue -= 1
        
        current_time += timedelta(minutes=time_slot_min)
                
        log_entry = {
            'date': current_time.strftime('%Y-%m-%d'),
            'time_slot': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'patients_in_queue': patients_in_queue
            # 'patients_in_general_room': general_room_occupancy,
            # 'patients_in_observation_room': observation_room_occupancy,
            # 'patients_in_surgery_room': surgery_room_occupancy
        }
                
        summary_data.append(log_entry)
       
    env.run(until=sim_duration_days * 24 * 60)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df
    

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
NUM_WAITING_ROOMS = 10

sim_duration_days = st.sidebar.slider("Simulation Duration (Days)", min_value=1, max_value=31, value=7)

#Select time slot min
time_slot_min  = st.sidebar.slider("Time Slot Intervals (in min)", min_value=0, max_value=60, value=15)

# start_date = st.date_input("Start Date", min_value=datetime(2024, 5, 1), max_value=datetime(2024, 5, 31), value=datetime(2024, 5, 2))
doctor_shifts = st.sidebar.slider("Doctor Shift (in hours)", min_value=3, max_value=8, value=5)
nurse_shifts = st.sidebar.slider("Nurse Shift (in hours)", min_value=7, max_value=14, value=8)

# start_datetime = datetime.now()
start_datetime = datetime(2024, 6, 1)


# patient_arrivals(start_datetime, time_slot_min, sim_duration_days)
simulate_hospital(sim_duration_days, start_datetime, time_slot_min)

# Simulate button
if st.sidebar.button("Simulate"):
    staff_month_schedule = simulate_staff_schedule(sim_duration_days, start_datetime)
    st.write("Doctors/Nurses Staff Schedule:")
    st.write(staff_month_schedule)
    
    #Load patient data
    patients_df = get_patients_df()
    st.write("Patients Arrivals Table")
    st.write(patients_df)

    
    summary_df = get_summary_df()
    st.write("Total Patients in Queue/Procedures Summary Schedule")
    st.write(summary_df)