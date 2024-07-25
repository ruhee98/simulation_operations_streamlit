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


patient_id = 0
patients_in_queue = 0
general_room_next_free = 0
surgery_room_next_free = 0
observation_room_next_free = 0

general_room_occupancy = 0
observation_room_occupancy = 0
surgery_room_occupancy = 0

patient_data = []
summary_data = []

prob_observation_after_general = 0.5
prob_stay_general_after_general = 0.2
prob_surgery_after_general = 0.2
prob_discharge_after_general = 0.1

prob_surgery_after_observation = 0.5
prob_discharge_after_observation = 0.5
env = simpy.Environment()
GENERAL_DURATION = st.sidebar.slider("Mean Duration of General Consultation", min_value=30, max_value=60, value=45)
SURGERY_DURATION = st.sidebar.slider("Mean Duration of Surgery Procedure", min_value=0, max_value=120, value=90)
OBSERVATION_DURATION =  st.sidebar.slider("Mean Duration of Testing/Observation Procedure", min_value=0, max_value=90, value=60)
NUM_GENERAL_ROOMS = st.sidebar.slider("Number of General Rooms", min_value=1, max_value=10, value=2)
NUM_SURGERY_ROOMS = st.sidebar.slider("Number of Surgery Rooms", min_value=1, max_value=10, value=2)
NUM_OBSERVATION_ROOMS = st.sidebar.slider("Number of Observation Rooms", min_value=1, max_value=10, value=2)
NUM_WAITING_ROOMS = 10

time_slot_min  = st.sidebar.slider("Time Slot Intervals (in min)", min_value=0, max_value=60, value=15)
sim_duration_days = st.sidebar.slider("Time Slot Intervals (in min)", min_value=0, max_value=31, value=7)
class Hospital:
    def __init__(self, env, num_general_rooms, num_observation_rooms, num_surgery_rooms, num_waiting_rooms):
        self.env = env
        self.general_room = simpy.Resource(env, capacity=num_general_rooms)  # Room A
        self.observation_room = simpy.Resource(env, capacity=num_observation_rooms)  # Room B
        self.surgery_room =  simpy.Resource(env, capacity=num_surgery_rooms)  # Room C
        self.waiting_room = simpy.Resource(env, capacity=num_waiting_rooms)
        # self.num_doctors = num_doctors
        # self.num_nurse_observation = num_nurses_observation
        # self.num_nurse_general = num_nurses_general
        # self.num_nurses_surgery = num_nurses_surgery

class Patient:
    def __init__(self, env, patient_id):
        self.env = env
        self.patient_id = patient_id
        self.procedure_type = None

    def set_priority(self):
        self.priority = random.choices(['Priority 1', 'Priority 2', 'Priority 3'], weights=[0.3, 0.5, 0.2], k=1)[0] 
        return self.priority
    
    def set_patient_outcome(self):
        if self.priority == 'Priority 1':
            self.procedure_type = 'Surgery'
        elif self.priority == 'Priority 2': 
            self.procedure_type = random.choices(['General', 'Observation'], [0.4, 0.6])[0]
        else:
            self.procedure_type = 'Home'
        return self.procedure_type

def assign_room(env, start_datetime, arrival_time_minutes, hospital, patient):

    global general_room_next_free, observation_room_next_free, surgery_room_next_free, patient_id, patients_in_queue

    # Assign procedure type and patient status (assuming these are fixed)
    procedure_type = random.choice(['general', 'observation', 'surgery'])
    
    assigned = False
    room_type = 'N/A'
    duration = GENERAL_DURATION
   
    arrival_datetime = start_datetime + timedelta(minutes=arrival_time_minutes % (24 * 60))
    delay = random.randint(30, 60)
    yield env.timeout(delay)
    
    with hospital.waiting_room.request() as request:
        yield request
        patient_id += 1
        patients_in_queue += 1

        treatment_start_time = arrival_datetime + timedelta(minutes=delay)
        wait_time = (treatment_start_time - arrival_datetime).total_seconds() / 60
        print(f"{patient_id} enters waiting room at {env.now} after waiting {wait_time}")
        patient = Patient(env, patient_id)

        duration = GENERAL_DURATION
        treatment_end_time = treatment_start_time + timedelta(minutes=duration)

        priority = patient.set_priority()
        procedure_type = patient.set_patient_outcome()

        room_request = None
        if procedure_type == 'General':
            room_request = hospital.general_room.request()
        elif procedure_type == 'Observation':
            room_request = hospital.observation_room.request()
        elif procedure_type == 'Surgery':
            room_request = hospital.surgery_room.request()

        if room_request:
            with room_request as req:
                yield req
                patients_in_queue -= 1
                print(f"Patient in queue after admission {procedure_type.lower()}:", patients_in_queue)

                if procedure_type == 'General':
                    assigned = True
                    room_type = 'General Room (A)'
                    if env.now < general_room_next_free:
                        yield env.timeout(max(0, general_room_next_free - env.now))
                    duration = GENERAL_DURATION
                    general_room_next_free = env.now + duration
                    log_patient_data(patient_id, arrival_datetime, priority, procedure_type, room_type, treatment_start_time, treatment_start_time + timedelta(minutes=duration), wait_time, duration)
        
                    yield env.timeout(duration)  
                    
                    rand_num = random.random()
                    if rand_num < prob_surgery_after_general:
                        patient.procedure_type = 'Surgery'
                        patient.priority = 'Priority 1'
                        log_patient_data(patient_id, arrival_datetime, patient.priority, patient.procedure_type, 'Surgery Room (C)', treatment_start_time, treatment_end_time, wait_time, duration)
                    elif rand_num < prob_surgery_after_general + prob_observation_after_general:
                        patient.procedure_type = 'Observation'
                        patient.priority = 'Priority 2'
                        log_patient_data(patient_id, arrival_datetime, patient.priority, patient.procedure_type, 'Observation Room (B)', treatment_start_time, treatment_end_time, wait_time, duration)
                    elif rand_num < prob_surgery_after_general + prob_observation_after_general + prob_stay_general_after_general:
                        patient.procedure_type = 'General'
                        patient.priority = 'Priority 2'
                        log_patient_data(patient_id, arrival_datetime, patient.priority, patient.procedure_type, 'General Room (A)', treatment_start_time, treatment_end_time, wait_time, duration)
                    else:
                        return patients_in_queue
       
                elif procedure_type == 'Observation':
                    room_type = 'Observation Room (B)'
                    if env.now < observation_room_next_free:
                        yield env.timeout(max(0, observation_room_next_free - env.now))
                    duration = OBSERVATION_DURATION
                    observation_room_next_free = env.now + duration
                    log_patient_data(patient.patient_id, arrival_datetime, priority, procedure_type, room_type, treatment_start_time, treatment_end_time, wait_time, duration)
                    yield env.timeout(duration)
                    
                    if random.random() < prob_surgery_after_observation:
                        patient.procedure_type = 'Surgery'
                        patient.priority = 'Priority 1'
                        log_patient_data(patient_id, arrival_datetime, patient.priority, patient.procedure_type, 'Surgery Room (C)', treatment_start_time, treatment_end_time, wait_time, duration)
                    else:
                        patient.procedure_type = 'None'
                        patient.priority = 'Priority 3'
                        log_patient_data(patient_id, arrival_datetime, patient.priority, patient.procedure_type, 'None', treatment_start_time, 'N/A', wait_time, 0)

                elif procedure_type == 'Surgery':
                    room_type = 'Surgery Room (C)'
                    log_patient_data(patient_id, arrival_datetime, patient.priority, patient.procedure_type, room_type, treatment_start_time, treatment_end_time, wait_time, duration)
                    if env.now < surgery_room_next_free:
                        yield env.timeout(max(0, surgery_room_next_free - env.now))
                    duration = SURGERY_DURATION
                    surgery_room_next_free = env.now + duration
                    yield env.timeout(duration)
                    
    return patients_in_queue

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

patient = Patient(env, patient_id)
hospital = Hospital(env, NUM_GENERAL_ROOMS, NUM_OBSERVATION_ROOMS, NUM_SURGERY_ROOMS, NUM_WAITING_ROOMS)
start_datetime = datetime.now()

def run_simulation():
    global general_room_occupancy, observation_room_occupancy, surgery_room_occupancy, patients_in_queue

    env = simpy.Environment()
    hospital = Hospital(env, NUM_GENERAL_ROOMS, NUM_OBSERVATION_ROOMS, NUM_SURGERY_ROOMS, NUM_WAITING_ROOMS)

    current_time = start_datetime

    while (current_time - start_datetime).days < sim_duration_days:
        arrival_time_minutes = random.randint(0, time_slot_min * 24)

        patient_process = env.process(assign_room(env, start_datetime, arrival_time_minutes, hospital, patient))
        env.run(until=patient_process)
        patients_in_queue = patient_process.value

        # env.process(assign_room(env, start_datetime, arrival_time_minutes, hospital, patient))
        # patients_in_queue += 1
        # print(f"Updated Patients in Queue: {patients_in_queue}")

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
                
        log_entry = {
            'date': current_time.strftime('%Y-%m-%d'),
            'time_slot': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'patients_in_queue': patients_in_queue,
            'patients_in_general_room': general_room_occupancy,
            'patients_in_observation_room': observation_room_occupancy,
            'patients_in_surgery_room': surgery_room_occupancy
        }
        current_time += timedelta(minutes=time_slot_min)
                    
        summary_data.append(log_entry)
    
    # env.run(until=sim_duration_days * 24 * 60)
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def get_summary_df():
    global summary_data
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def get_patients_df():
    global patient_data
    patient_df = pd.DataFrame(patient_data)
    return patient_df

run_simulation()


if st.sidebar.button("Simulate"):
    # staff_month_schedule = simulate_staff_schedule(sim_duration_days, start_datetime)
    # st.write("Doctors/Nurses Staff Schedule:")
    # st.write(staff_month_schedule)
    
    #Load patient data
    patients_df = get_patients_df()
    st.write("Patients Arrivals Table")
    st.write(patients_df)
    
    summary_df = get_summary_df()
    st.write("Total Patients in Queue/Procedures Summary Schedule")
    st.write(summary_df)